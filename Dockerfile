FROM tensorflow/tensorflow:latest-gpu-py3

RUN apt-get update
RUN apt-get -y install libsm6 libxrender1 libfontconfig1 libxext6
RUN pip install opencv-python filterpy numba

add ./ /assigment

WORKDIR /assigment

ENTRYPOINT ["python", "face_detect_track.py"]