#!/usr/bin/env bash


if [ ! -f weights.pkl ]; then
    echo "downloading weights" 
    wget https://www.cs.cmu.edu/%7Epeiyunh/tiny/hr_res101.mat
    python utils/matconvnet_hr101_to_pickle.py \
            --matlab_model_path hr_res101.mat \
            --weight_file_path  weights.pkl 
fi

if [ ! -f input/worlds-largest-selfie.jpg ]; then
   echo "worlds-largest-selfie.jpg not in directory"
else
python face_detect_track.py --input input/worlds-largest-selfie.jpg --same_colour
fi

if [ ! -f input/oscars.mp4 ]; then
   echo "oscars.mp4 not in directory"
else
    python face_detect_track.py --input input/oscars.mp4 --output output/oscars/ --mode video

    cat output/oscars/*.jpg | ffmpeg -f image2pipe -framerate 29.97  -vcodec mjpeg -i - -vcodec libx264 -b:v 2M -maxrate 2M -bufsize 2M  output/oscars_tracked.mp4

    ffmpeg -i output/oscars_tracked.mp4 -i oscars.mp4 -c copy -map 0:0 -map 1:1 -shortest output/oscars_tracked_audio.mp4 \
    && mv output/oscars_tracked_audio.mp4 output/oscars_tracked.mp4

fi

