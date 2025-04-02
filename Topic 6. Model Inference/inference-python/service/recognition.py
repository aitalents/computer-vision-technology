""" Build a recognition service from infrastructure models"""

from dataclasses import asdict
from typing import List
import logging

import numpy as np

from service.interface import ServicePredictionDataModel, RecognitionServiceInterface


class DummyRecognitionService(RecognitionServiceInterface):

    def recognize_image(self, image: np.ndarray) -> List[ServicePredictionDataModel]:
        """Function to inference image on detector and classifier

        Args:
            image (np.ndarray): [description]

        Returns:
            List[ServicePrediction]: [description]
        """
        logging.info("Start recognizing image ... ")
        results = []

        detected_contours = self.detector.detect_contours(image=image)
        logging.info("Classify detected contours ...")
        for detected_contour in detected_contours:
            x_min, y_min, x_max, y_max = detected_contour.predicted_contour.rectangle
            cropped_image = image[y_min:y_max, x_min:x_max]

            classifier_predictions = self.classifier.predict(cropped_image)
            pred = ServicePredictionDataModel(
                predicted_class=classifier_predictions.predicted_class,
                class_probability=classifier_predictions.probability,
                contour_probability=detected_contour.contour_probability,
                x_min=x_min,
                y_min=y_min,
                width=x_max-x_min,
                height=y_max-y_min
            )
            results.append(pred)
        return results

if __name__ == "__main__":

    from infrastructure.detection.model import DummyDetector, Yolov5Detector
    from infrastructure.classification.model import DummyClassfier

    # Init dummy models
    #detector = DummyDetector("path/to/model/file.model")
    detector = Yolov5Detector()
    classifier = DummyClassfier("path/to/model/file.model")

    # Init recognition service
    recognition_service = DummyRecognitionService(detector, classifier)

    # create random image
    imarray = np.random.rand(1000,1000,3) * 255

    # get dummy predictions
    predictions = recognition_service.recognize_image(image=imarray)
    for i, pred in enumerate(predictions):
        print(f"Predictions {i}: \n {asdict(pred)} \n")
