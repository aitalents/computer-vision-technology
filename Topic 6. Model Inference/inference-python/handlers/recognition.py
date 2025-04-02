from abc import ABC, abstractmethod
from typing import Tuple, List
import logging

import numpy as np
from marshmallow import ValidationError

from service.recognition import RecognitionServiceInterface
from service.image_loader import ImageLoader
from handlers.schemas import RecognitionResultsSchema, RecognitionRequestSchema


class RecognitionHandler(ABC):
    def __init__(self, recognition_service: RecognitionServiceInterface, image_loader: ImageLoader):
        self.recognition_service = recognition_service
        self.image_loader = image_loader
        self.response_scheme = RecognitionRequestSchema()
        self.result_scheme = RecognitionResultsSchema()

    @abstractmethod
    def handle(self, body: str):
        pass

    @abstractmethod
    def _load_image(self, url: str) -> np.ndarray:
        pass

    def deserialize_body(self, body: str) -> Tuple[RecognitionRequestSchema, str]:
        logging.info("Deserializing body...")

        body_schema = None
        errs = None
        try:
            body_schema = self.response_scheme.load(body)
        except ValidationError as err:
            errs = err.messages
        return body_schema, errs

    def serialize_answer(self, result, task_id) -> str:
        logging.info("Serializing body...")
        return self.result_scheme.dump({"task_id": task_id, "recognitions": result})


class DummyRecognitionHandler(RecognitionHandler):
    def __init__(self, recognition_service: RecognitionServiceInterface, image_loader: ImageLoader):
        super().__init__(recognition_service, image_loader)

    def handle(self, body: str) -> List:
        """
        Takes response body from Queue
        - deserialize and validate body fields
        - make service calls, Load image and extract products
        - serialize results
        - return results to transport layer
        :param body:
        :return:
        """
        results = []
        body_schema, errs = self.deserialize_body(body)
        if body_schema:
            task_id = body_schema['task_id']
            # call image loader
            image = self._load_image(body_schema['image_url'])
            rec_results = self.recognition_service.recognize_image(image=image)

            results = [self.serialize_answer(recognition, task_id) for recognition in rec_results]
        else:
            results = [errs]
        return results

    def _load_image(self, url: str) -> np.ndarray:
        """
        Call an image loader
        :param url:
        :return:
        """
        image_array = None
        try:
            img_bytes = self.image_loader.load_image(url=url)
            image_array = self.image_loader.image_as_array(img_bytes)
            logging.info("Image succefully loaded")
        except Exception as err:
            print(err)
        return image_array
