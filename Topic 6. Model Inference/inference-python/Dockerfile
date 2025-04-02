#FROM python:3.9.9
FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime
ARG DEBIAN_FRONTEND=noninteractive

ARG APP_DIR=/app
WORKDIR "$APP_DIR"

COPY requirements.txt $APP_DIR/

RUN apt-get update -y && \
	apt-get install -y libsm6 libxrender1 libfontconfig1 && \
    apt-get install -y libxext6 libgl1-mesa-glx ffmpeg

RUN pip install -U pip && \
    pip install -U setuptools && \
    pip install -r requirements.txt

# COPY . $APP_DIR/
# ENTRYPOINT ["python", "app.py"]
ENTRYPOINT ["bash"]
