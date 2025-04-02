""" Build a recognition service from infrastructure models"""

from abc import abstractmethod, ABC
from dataclasses import dataclass
from typing import List

import numpy as np

from infrastructure.detection.model import Detector
from infrastructure.classification.model import Classifier


@dataclass
class ServicePredictionDataModel:
    predicted_class: str
    class_probability: float
    contour_probability: float
    x_min: int
    y_min: int
    width: int
    height: int


class RecognitionServiceInterface(ABC):
    def __init__(self, detector: Detector, classifier: Classifier):
        self.detector = detector
        self.classifier = classifier

    @abstractmethod
    def recognize_image(self, image: np.ndarray) -> List[ServicePredictionDataModel]:
        """Function to inference image on detector and classifier

        Args:
            image (np.ndarray): [description]

        Returns:
            List[ServicePrediction]: [description]
        """
        results = []
        return results

