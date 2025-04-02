from abc import ABC, abstractmethod
from typing import List
from dataclasses import dataclass
import logging

import numpy as np
import torch

from infrastructure.contour import Contour


@dataclass
class DetectorPrediction:
    predicted_contour: Contour
    contour_probability: float


class Detector(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def detect_contours(self, image: np.ndarray) -> List[DetectorPrediction]:
        """
        Return boxes of all detected contours from image.
        :param image: np.ndarray RGB image
        :return: List[Contour]
        """
        pass


class DummyDetector(Detector):

    def __init__(self, detection_model_path: str):
        logging.info('Loading Detector')
        self.model_path = detection_model_path

    def detect_contours(self, image: np.ndarray) -> List[DetectorPrediction]:
        # create square 100x100 pixels contour
        dummy_contour = Contour(bounding_rect=(0, 0, 100, 100))
        prediction = DetectorPrediction(predicted_contour=dummy_contour, contour_probability=0.98)
        return [prediction]


class Yolov5Detector(Detector):

    def __init__(self):
        logging.info('Loading Detector')
        self.model = self._load_model()

    def _load_model(self):
        model = torch.hub.load('ultralytics/yolov5', 'yolov5s', force_reload=True)
        return model

    def detect_contours(self, image: np.ndarray) -> List[DetectorPrediction]:
        predictions = []

        raw_results = self.model(image)
        df_results = raw_results.pandas().xyxy[0]
        df_persons = df_results[df_results["class"]==0]
        for i, row in df_persons.iterrows():
            contour = Contour(
                    bounding_rect=(int(row.xmin), int(row.ymin), int(row.xmax-row.xmin), int(row.ymax-row.ymin))
                    )
            prediction = DetectorPrediction(predicted_contour=contour, contour_probability=row.confidence)
            predictions.append(prediction)
        return predictions

