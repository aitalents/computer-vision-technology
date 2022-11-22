FROM python:3.9
WORKDIR /app

ENV PYTHONPATH="${PYTHONPATH}:${WORKDIR}"

COPY requirements.txt $WORKDIR

RUN pip install -r requirements.txt && \
    apt-get update && \
    apt install ffmpeg libsm6 libxext6 -y

CMD "bash"