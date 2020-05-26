# Before build this image make sure that:
# 1) nvidia-docker2 is installed (https://github.com/NVIDIA/nvidia-docker)
# 2) NVIDIA proprietary driver (version 410+) is installed
# 4) nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb in this directory

# Example commands to build image and run container:
# docker build -t fast_yolo_image:latest -f Dockerfile .

# docker run -d -t --runtime=nvidia \
#       --name fast_yolo \
#       -e DISPLAY=:0 -e QT_X11_NO_MITSHM=1 \
#       --privileged \
#       -v /dev/bus/usb:/dev/bus/usb \
#       -v /tmp/.X11-unix:/tmp/.X11-unix \
#       -i fast_yolo_image:latest
# docker exec -it fast_yolo /bin/bash

FROM nvidia/cuda:10.2-cudnn7-devel-ubuntu18.04

#GLOBAL VARS

ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV LC_ALL C.UTF-8

#Common install
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        sudo vim wget tar git unzip \
        pkg-config \
        libsm6 libxext6 libxrender-dev \
        python3-dev python3-setuptools python3-pip 

RUN python3 -m pip install --upgrade pip

# TensorRT for python
COPY ./nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb /root/
RUN dpkg -i /root/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb && \
    apt-key add /var/nv-tensorrt-repo-cuda10.2-trt7.0.0.11-ga-20191216/7fa2af80.pub && \
    apt-get update && \
    apt install -y tensorrt python3-libnvinfer-dev uff-converter-tf

RUN pip3 install \
        torch torchvision \
        numpy scikit-learn scipy \
	onnx opencv-python

RUN rm -rf /root/*

RUN git clone https://github.com/VolkovAK/FastYoloInference.git /root/FastYoloInference

WORKDIR /root/FastYoloInference

