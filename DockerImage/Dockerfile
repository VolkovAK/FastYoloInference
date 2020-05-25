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

ARG OPENCV_VERSION=4.3.0
ENV NVIDIA_VISIBLE_DEVICES \
    ${NVIDIA_VISIBLE_DEVICES:-all}
ENV NVIDIA_DRIVER_CAPABILITIES \
    ${NVIDIA_DRIVER_CAPABILITIES:+$NVIDIA_DRIVER_CAPABILITIES,}graphics
ENV BUILD_DIR /root/build
ENV LC_ALL C.UTF-8
ENV DEBIAN_FRONTEND noninteractive

#Common install
RUN apt-get update
RUN apt-get install -y --no-install-recommends \
        sudo vim wget tar git automake libtool-bin libconfig++-dev \
        ffmpeg \
        cmake \
        build-essential \
        pkg-config \
        libboost-all-dev \
        libatlas-base-dev \
        libgflags-dev \
        libgoogle-glog-dev \
        libhdf5-serial-dev \
        libleveldb-dev \
        liblmdb-dev \
        libavdevice-dev \
        libavcodec-dev \
        libavformat-dev \
        libswscale-dev \
        libavutil-dev \
        libsnappy-dev \
        libblas-dev \
        liblapack-dev \
        libtbb-dev libtbb2 \
        libpng-dev \
        libjpeg-dev \
        libopenexr-dev \
        libtiff-dev \
        libwebp-dev \
        libxvidcore-dev libx264-dev libx265-dev

RUN apt-get install -y --no-install-recommends \
        libgtk2.0-dev \
        libmatio-dev \
        libprotobuf-dev \
        protobuf-compiler \
        libtool-bin \
        libcanberra-gtk-module \
        libxml++2.6-dev libgeographic-dev uuid-dev \
        libcurl4-gnutls-dev zlib1g-dev \
        libsdl2-dev libsdl2-ttf-dev \
        expect-dev unzip apt-utils \
        libgstreamer-plugins-base1.0-dev \
        libgstreamer1.0-dev \
        libgtk-3-dev \
        python3-dev python3-setuptools python3-pip 

# set your timezone
RUN ln -fs /usr/share/zoneinfo/Europe/Moscow /etc/localtime
RUN dpkg-reconfigure --frontend noninteractive tzdata

RUN pip3 install --upgrade pip && pip3 install numpy scipy
RUN rm -rf /var/lib/apt/lists/*


# #OpenCV (Open Source Computer Vision Library) - builds too long, but you can try. Did not give consistent boost
WORKDIR ${BUILD_DIR}
RUN mkdir -p ${BUILD_DIR} && \
    git clone --branch ${OPENCV_VERSION} https://github.com/opencv/opencv.git ${BUILD_DIR}/opencv; \
    git clone --branch ${OPENCV_VERSION} https://github.com/opencv/opencv_contrib.git ${BUILD_DIR}/opencv_contrib

RUN cd opencv && \
    mkdir build && cd build && cmake \
        -D CMAKE_BUILD_TYPE=RELEASE \
        -D OPENCV_EXTRA_MODULES_PATH=${BUILD_DIR}/opencv_contrib/modules \
        -D BUILD_TIFF=ON \
        -D WITH_CUDA=ON \
        -D WITH_CUBLAS=ON \
        -D WITH_CUDNN=ON \
        -D WITH_NVCUVID=ON \
        -D WITH_FFMPEG=ON \
        -D WITH_IPP=ON \
        -D WITH_TBB=ON \
        -D WITH_OPENMP=ON \
        -D WITH_LAPACK=ON \
        -D WITH_CSTRIPES=ON \
        -D WITH_TIFF=ON \
        -D WITH_V4L=ON \
        -D WITH_OPENCL=ON \
        -D BUILD_opencv_apps=OFF \
        -D BUILD_DOCS=OFF \
        -D BUILD_EXAMPLES=OFF \
        -D BUILD_PERF_TESTS=OFF \
        -D BUILD_TESTS=OFF \
        -D BUILD_TBB=ON \
        -D ENABLE_PRECOMPILED_HEADERS=OFF \
        -D ENABLE_FAST_MATH=ON \
        -D CUDA_FAST_MATH=ON \
        -D CMAKE_INSTALL_PREFIX=/usr/local \
        -D OPENCV_GENERATE_PKGCONFIG=ON \
        -D PYTHON_DEFAULT_EXECUTABLE=/usr/bin/python3 \
        -D BUILD_NEW_PYTHON_SUPPORT=ON \
        -D BUILD_opencv_python3=ON \
        -D HAVE_opencv_python3=ON .. && \
    make -j $(nproc) && \
    make install && \
    cd ../.. && \
    rm -rf opencv && rm -rf opencv_contrib


# TensorRT for python
ADD ./packages/ /root/
RUN dpkg -i /root/nv-tensorrt-repo-ubuntu1804-cuda10.2-trt7.0.0.11-ga-20191216_1-1_amd64.deb && \
    apt-key add /var/nv-tensorrt-repo-cuda10.2-trt7.0.0.11-ga-20191216/7fa2af80.pub && \
    apt-get update && \
    apt install -y tensorrt python3-libnvinfer-dev uff-converter-tf

RUN pip3 install torch torchvision numpy scikit-learn scipy ffmpeg-python onnx psutil onnxruntime skl2onnx xlrd

RUN rm -rf /root/*

RUN git clone https://github.com/VolkovAK/FastYoloInference.git 

WORKDIR /root/FastYoloInference

