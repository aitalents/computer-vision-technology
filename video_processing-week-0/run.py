import cv2
import numpy as np
import tritonclient.grpc as grpcclient

from streamer import FrameVideoIterator
from utils import batch_preprocessing, plot_detections


triton_url = "10.10.67.125:8001"

client = grpcclient.InferenceServerClient(url=triton_url, verbose=False)


def triton_inference(img: np.ndarray):
    batch = batch_preprocessing([img])

    inputs = []
    inputs.append(grpcclient.InferInput("images", batch.shape, "FP32"))
    inputs[0].set_data_from_numpy(batch)

    outputs = []
    outputs.append(grpcclient.InferRequestedOutput("output_post"))

    results = client.infer("yolov5s_pipe", inputs, outputs=outputs)
    return results.as_numpy("output_post")



if __name__ == "__main__":
    src = "rtsp://10.10.67.125:8554/test"
    save_path = "./plotted/"
    video_iterator = FrameVideoIterator(src)
    stop = 10

    for i, frame in video_iterator:
        if i <= stop:
            detections = triton_inference(frame)
            print(f"Detections on frame {i}: {detections}")
            plot_frame = plot_detections(frame, detections)
            cv2.imwrite(f"{save_path}frame_{i}.jpg", plot_frame[:,:,::-1])
        else:
            break
