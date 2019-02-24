# Tracking Tiny Faces

Tracking multiple tiny faces in an image or video sequence in tensorflow. No tracking recovery of a particular face, so a new id is assigned to a face if the tracking fails.

# Dependencies

- tensorflow, opencv-python, filterpy, numba

```
pip install -r requirements.txt
```

- ffmpeg (to convert image sequence to video)

```
sudo apt-get install ffmpeg
```

# Pretrained model weights

Tiny face model weights were downloaded from [ResNet101-based pretrained model(hr_res101.mat)](https://www.cs.cmu.edu/%7Epeiyunh/tiny/hr_res101.mat) and converted to a format readable in python with the script `matconvnet_hr101_to_pickle`

```
wget https://www.cs.cmu.edu/%7Epeiyunh/tiny/hr_res101.mat
python utils/matconvnet_hr101_to_pickle.py \
        --matlab_model_path hr_res101.mat \
        --weight_file_path  weights.pkl 

```


# Usage


## Face detection and tracking


Face detection only on an image or directory of images
```
python face_detect_track.py --input input/worlds-largest-selfie.jpg  --same_colour
```

Face detection and tracking on a video
```
python face_detect_track.py --input input/oscars.mp4 --output output/oscars/ --mode video
```

This generates an images sequence with bounding boxes overlays.

To convert to a video, run

```
cat output/oscars/*.jpg | ffmpeg -f image2pipe -framerate 29.97  -vcodec mjpeg -i - -vcodec libx264 -b:v 3M -maxrate 3M -bufsize 2M  output/oscars_tracked.mp4


```
and to add back the audio track

```
ffmpeg -i output/oscars_tracked.mp4 -i oscars.mp4 -c copy -map 0:0 -map 1:1 -shortest output/oscars_tracked_audio.mp4 \
&& mv output/oscars_tracked_audio.mp4 output/oscars_tracked.mp4
```

## Docker image

You can also use a docker image. First, build the image with

```
docker build -t face_detect_track .
```

Then run with (requires [nvidia-docker](https://github.com/NVIDIA/nvidia-docker))

```
nvidia-docker run -it face_detect_track --help
```

## Run all

run the script file `run.sh`

## Output


![](output/oscars_tracked.gif)


# Disclaimer/Acknowledgment

- Tiny face detection code was adapted from [cydonia999](https://github.com/cydonia999/Tiny_Faces_in_Tensorflow)
- Tracking code was adapted from [abewley](https://github.com/abewley/sort)