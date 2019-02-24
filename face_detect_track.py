# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import os
import pickle
import time
from argparse import ArgumentParser

import cv2
import numpy as np
import pylab as pl
import tensorflow as tf
import tiny_face_model
from scipy.special import expit
from sort import Sort

MAX_INPUT_DIM = 5000.0
COLOURS = 255 * np.random.rand(100, 3)


def overlay_bounding_box(image, box, display_id=False, random_color=True):
    box = np.array(box, dtype=np.int)
    bb_id = box[4]
    h = box[2] - box[0]
    w = box[3] - box[1]
    font_scale = min(0.6, 0.1 + w * h / 10000)
    font_thickness = int(1 + round(font_scale))

    if random_color:
        colour = COLOURS[bb_id % len(COLOURS), :]
    else:
        colour = (255, 255, 0)

    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), (0, 0, 0),
                  font_thickness + 1)
    cv2.rectangle(image, (box[0], box[1]), (box[2], box[3]), colour,
                  font_thickness)

    if display_id:
        text = "{}".format(bb_id)
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
        cv2.putText(image, text, (box[0] + 2, box[1] + text_size[0][1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 1 + font_thickness)
        cv2.putText(image, text, (box[0] + 2, box[1] + text_size[0][1] + 2),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, colour, font_thickness)


class FaceDetector:
    def __init__(self, weight_file_path, session, prob_threshold=0.5, nms_threshold=0.1):
        self.sess = session
        self.prob_thresh = prob_threshold
        self.nms_thresh = nms_threshold

        # Create the tiny face model which weights are loaded from a pretrained model.
        self.model = tiny_face_model.Model(weight_file_path)

        # placeholder of input images. Currently batch size of one is supported.
        self.x = tf.placeholder(tf.float32, [1, None, None, 3])  # n, h, w, c

        self.score_final = self.model.tiny_face(self.x)

        # intialise variables
        self.sess.run(tf.global_variables_initializer())

        # Load an average image and clusters(reference boxes of templates).
        with open(weight_file_path, "rb") as f:
            _, mat_params_dict = pickle.load(f)

        self.average_image = self.model.get_data_by_key("average_image")
        self.clusters = self.model.get_data_by_key("clusters")
        self.clusters_h = self.clusters[:, 3] - self.clusters[:, 1] + 1
        self.clusters_w = self.clusters[:, 2] - self.clusters[:, 0] + 1
        self.normal_idx = np.where(self.clusters[:, 4] == 1)

    def _calc_scales(self, image):
        raw_h, raw_w = image.shape[0], image.shape[1]
        min_scale = min(np.floor(np.log2(np.max(self.clusters_w[self.normal_idx] / raw_w))),
                        np.floor(np.log2(np.max(self.clusters_h[self.normal_idx] / raw_h))))
        max_scale = min(1.0, -np.log2(max(raw_h, raw_w) / MAX_INPUT_DIM))
        scales_down = pl.frange(min_scale, 0, 1.)
        scales_up = pl.frange(0.5, max_scale, 0.5)
        scales_pow = np.hstack((scales_down, scales_up))
        scales = np.power(2.0, scales_pow)
        return scales

    def _calc_bounding_boxes(self, scale, prob_cls_tf, score_cls_tf, score_reg_tf):
        # threshold for detection
        _, fy, fx, fc = np.where(prob_cls_tf > self.prob_thresh)

        # interpret heatmap into bounding boxes
        cy = fy * 8 - 1
        cx = fx * 8 - 1
        ch = self.clusters[fc, 3] - self.clusters[fc, 1] + 1
        cw = self.clusters[fc, 2] - self.clusters[fc, 0] + 1

        # extract bounding box refinement
        Nt = self.clusters.shape[0]
        tx = score_reg_tf[0, :, :, 0:Nt]
        ty = score_reg_tf[0, :, :, Nt:2 * Nt]
        tw = score_reg_tf[0, :, :, 2 * Nt:3 * Nt]
        th = score_reg_tf[0, :, :, 3 * Nt:4 * Nt]

        # refine bounding boxes
        dcx = cw * tx[fy, fx, fc]
        dcy = ch * ty[fy, fx, fc]
        rcx = cx + dcx
        rcy = cy + dcy
        rcw = cw * np.exp(tw[fy, fx, fc])
        rch = ch * np.exp(th[fy, fx, fc])

        scores = score_cls_tf[0, fy, fx, fc]
        tmp_bboxes = np.vstack((rcx - rcw / 2, rcy - rch / 2, rcx + rcw / 2, rcy + rch / 2))
        tmp_bboxes = np.vstack((tmp_bboxes / scale, scores))
        tmp_bboxes = tmp_bboxes.transpose()
        return tmp_bboxes

    def detect_faces(self, image, filename):
        image_f = image.astype(np.float32)

        scales = self._calc_scales(image)
        start = time.time()

        # initialize output
        bboxes = np.empty(shape=(0, 5))

        # process input at different scales
        for s in scales:
            print("Processing {} at scale {:.4f}".format(filename, s))
            img = cv2.resize(image_f, (0, 0), fx=s, fy=s, interpolation=cv2.INTER_LINEAR)
            img = img - self.average_image
            img = img[np.newaxis, :]

            # we don't run every template on every scale ids of templates to ignore
            tids = list(range(4, 12)) + ([] if s <= 1.0 else list(range(18, 25)))
            ignored_tids = list(set(range(0, self.clusters.shape[0])) - set(tids))

            # run through the network
            score_final_tf = self.sess.run(self.score_final, feed_dict={self.x: img})

            # collect scores
            score_cls_tf, score_reg_tf = score_final_tf[:, :, :, :25], score_final_tf[:, :, :, 25:125]
            prob_cls_tf = expit(score_cls_tf)
            prob_cls_tf[0, :, :, ignored_tids] = 0.0

            tmp_bboxes = self._calc_bounding_boxes(s, prob_cls_tf, score_cls_tf, score_reg_tf)
            bboxes = np.vstack((bboxes, tmp_bboxes))

        refind_idx = tf.image.non_max_suppression(tf.convert_to_tensor(bboxes[:, :4], dtype=tf.float32),
                                                  tf.convert_to_tensor(bboxes[:, 4], dtype=tf.float32),
                                                  max_output_size=bboxes.shape[0], iou_threshold=self.nms_thresh)
        refind_idx = self.sess.run(refind_idx)
        refined_bboxes = bboxes[refind_idx]
        print("detection time {:.2f} secs for {}".format(time.time() - start, filename))

        return refined_bboxes


def main():
    parser = ArgumentParser()
    parser.add_argument("--weight_file_path", default='weights.pkl', help='path to model weights')
    parser.add_argument("--output_dir", default='./output')
    parser.add_argument("--input", default='./images/')
    parser.add_argument("--prob_thresh", default=0.7, type=float)
    parser.add_argument("--nms_thresh", default=0.1, type=float)
    parser.add_argument("--max_trk_age", default=50, type=int,
                        help='delete face tracker if lost tracking for more than max_trk_age frames')
    parser.add_argument("--display_id", action='store_true')
    parser.add_argument("--same_colour", action='store_false')
    parser.add_argument("--mode", default='image', choices=['image', 'image_seq', 'video'],
                        help='image (detection only): image file or directory of images.'
                             'image_seq (detection + tracking): directory of images.'
                             'video (detection + tracking): video file. Saves to image sequence.')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    img_exts = ('*.png', '*.gif', '*.jpg', '*.jpeg')

    do_track = False
    filenames = []

    if args.mode == 'image_seq':
        do_track = True
        for ext in img_exts:
            filenames.extend(glob.glob(os.path.join(args.input, ext)))

    elif args.mode == 'video':
        do_track = True

    else:
        if os.path.isdir(args.input):
            for ext in img_exts:
                filenames.extend(glob.glob(os.path.join(args.input, ext)))
        else:
            if os.path.exists(args.input):
                filenames = [args.input]

            else:
                print('%s does not exist' % args.input)
                exit(1)

    filenames.sort()

    with tf.Session() as sess:

        # initialise face detector
        detector = FaceDetector(args.weight_file_path, session=sess)

        # initialise multiple object tracker
        tracker = Sort(max_age=args.max_trk_age, min_hits=5)

        def process_frame(image, fname):

            # detect faces
            faces_bboxes = detector.detect_faces(image, fname)

            # update multiple face trackers
            if do_track:
                start = time.time()
                faces_bboxes = tracker.update(faces_bboxes[:, :4].astype(np.int))
                print("tracking update time {:.2f} secs".format(time.time() - start))

            # overlay face detections
            for box in faces_bboxes:
                overlay_bounding_box(image, box, args.display_id, args.same_colour)

            # overlay face count
            text = 'Number of faces: {}'.format(len(faces_bboxes))
            cv2.putText(image, text, (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 0, 0), 2)
            cv2.putText(image, text, (10, image.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (0, 255, 0), 1)
            return image

        if args.mode == 'video':
            vid_cap = cv2.VideoCapture(args.input)

            # num_frames = int(vid_cap.get(cv2.CAP_PROP_FRAME_COUNT))

            basename = os.path.basename(args.input).split('.')[0]
            frame_count = 0

            while True:
                success, image = vid_cap.read()
                if success:
                    fname = basename + '%04d.jpg' % frame_count
                    frame_count += 1
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    image = process_frame(image, fname)
                    # save image with bounding boxes
                    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                    cv2.imwrite(os.path.join(args.output_dir, fname), image)
                else:
                    break

        # image or image sequence
        else:
            for frame_count, filename in enumerate(filenames):
                fname, ext = os.path.splitext(os.path.basename(filename))
                image = cv2.imread(filename)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = process_frame(image, fname)

                # save image with bounding boxes
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                cv2.imwrite(os.path.join(args.output_dir, fname + '_face_detection' + ext), image)


if __name__ == '__main__':
    main()
