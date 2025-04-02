from dataclasses import dataclass
from abc import ABC, abstractmethod

import numpy as np


@dataclass
class ClassifierPrediction:
    predicted_class: str
    probability: float


class Classifier(ABC):

    @abstractmethod
    def __init__(self, model_path: str):
        pass

    @abstractmethod
    def predict(self, image: np.ndarray) -> ClassifierPrediction:
        """Predict a class of an image

        Args:
            image (np.ndarray): [description]

        Returns:
            ClassifierPrediction:
        """
        pass


class DummyClassfier(Classifier):

    def __init__(self, model_path: str):
        super().__init__(model_path)

    def predict(self, image: np.ndarray) -> ClassifierPrediction:
        prediction = ClassifierPrediction(predicted_class="Human", probability=0.99)
        return prediction
