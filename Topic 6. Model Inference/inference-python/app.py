import json
import logging
import time

from flask import Flask, request, Response

from infrastructure.detection.model import DummyDetector, Yolov5Detector
from infrastructure.classification.model import DummyClassfier

from service.image_loader import RequestImageLoader
from service.recognition import DummyRecognitionService

from handlers.recognition import DummyRecognitionHandler

from config import AppConfig


config = AppConfig()

# Init dummy models
#detector = DummyDetector(config.DETECTOR_MODEL_PATH)
detector  = Yolov5Detector()
classifier = DummyClassfier(config.CLASSIFIER_MODEL_PATH)

# Init recognition service
recognition_service = DummyRecognitionService(detector, classifier)
image_loader = RequestImageLoader()

# Init handler
handler = DummyRecognitionHandler(recognition_service, image_loader)

app = Flask(config.APP_NAME)


@app.route('/healthcheck')
def healthcheck():
    return Response(status=200)


@app.route('/recognize', methods=['POST'])
def recognize():

    body = request.json
    time_start = time.time()

    result = handler.handle(body)
    elapsed_time = time.time() - time_start

    logging.info({'message': f'Image processing on CV completed in {elapsed_time} seconds',
                  'elapsed_time': elapsed_time,
                  'state': 'processing_completed'})

    return Response(response=json.dumps(result), status=200, mimetype="application/json")


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=config.PORT, debug=config.DEBUG)
