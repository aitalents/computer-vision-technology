FROM python:3.9

WORKDIR /src

COPY requirements.txt $WORKDIR

RUN pip install -r requirements.txt && \
	apt-get update && \
    apt-get install ffmpeg libsm6 libxext6  -y

CMD ["jupyter", "notebook", "--port=8888", "--no-browser", "--ip=0.0.0.0", "--NotebookApp.token=''", "--allow-root"]
