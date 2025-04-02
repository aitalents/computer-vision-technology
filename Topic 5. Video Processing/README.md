### Find rtsp stream or run by yourself
[rtspstramer repo](https://github.com/electriclizard/rtspstreamer)

### Run the triton inference server with yolo onnx model
[triton repo from mlops course](https://github.com/aitalents/mlops-course/tree/main/week03-triton)

##### starts with
`docker run -ti --gpus device=1 --rm --shm-size=12G -p 8000:8000 -p 8001:8001 -p 8002:8002 -v $PWD/models/:/models --name itmo-triton-server triton-itmo tritonserver --model-repository=/models`

### And `python run.py` for inference rtsp stream on triton
