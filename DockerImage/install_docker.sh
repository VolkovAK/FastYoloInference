#/bin/bash

# check if nvidia driver is installed
if ! nvidia-smi>/dev/null; then
    echo "Please, install nvidia driver! For more info check https://github.com/VolkovAK/FastYoloInference"
    echo "Abort"
    exit
fi

# get supported cuda version from nvidia driver info
cuda=$(nvidia-smi -q --display="COMPUTE" | grep "CUDA" | grep -o "[0-9\.]*")

if [[ "$cuda" = "10.0" || "$cuda" = "10.1" ]]; then
    dockerfile=$"Dockerfile.10.0"
elif [ "$cuda" = "10.2" ]; then
    dockerfile=$"Dockerfile.10.2"
else
    echo "Please, install nvidia driver with support cuda version >= 10.0"
    echo "Abort"
    exit
fi

docker build -t fast_yolo_image:latest -f $dockerfile .    
