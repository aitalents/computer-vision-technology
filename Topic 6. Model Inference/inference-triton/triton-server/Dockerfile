FROM nvcr.io/nvidia/tritonserver:22.04-py3

ARG DEBIAN_FRONTEND=noninteractive

RUN wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin && \
    mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600 && \
    apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub && \
    add-apt-repository "deb https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/ /" && \
    pip install torch==1.10.0+cpu torchvision==0.11.0+cpu -f https://download.pytorch.org/whl/torch_stable.html && \
    apt-get update && apt-get clean && apt-get install -fy && dpkg-reconfigure debconf

RUN apt install ffmpeg libsm6 libxext6  -y && \
    # pip install opencv-python
    rm -rf /var/lib/{apt,dpkg,cache,log}/
