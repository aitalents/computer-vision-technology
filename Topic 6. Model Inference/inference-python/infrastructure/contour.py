from enum import Enum
from typing import Optional, Tuple

import cv2
import numpy as np


class Contour:
    """
    >>> Contour(bounding_rect=(1, 2, 3, 4)).area
    12
    >>> Contour(bounding_rect=(1, 2, 3, 4)).width
    3
    >>> Contour(bounding_rect=(1, 2, 3, 4)).height
    4
    >>> Contour(bounding_rect=(1, 2, 3, 4)).x_min
    1
    >>> Contour(bounding_rect=(1, 2, 3, 4)).x_max
    4
    >>> Contour(bounding_rect=(1, 2, 3, 4)).y_min
    2
    >>> Contour(bounding_rect=(1, 2, 3, 4)).y_max
    6
    >>> Contour(bounding_rect=(1, 2, 3, 4)).rectangle
    (1, 2, 4, 6)
    """

    def __init__(self,
                 contour: Optional[np.ndarray] = None,
                 bounding_rect: Optional[Tuple[int, int, int, int]] = None,
                 ):
        """
        :param contour: OpenCV-like contour with many points
        :param bounding_rect: tuple with four elements: x, y, width, height
        """
        if contour is None:
            if not len(bounding_rect) or any(isinstance(x, float) for x in bounding_rect):
                raise ValueError(f'bounding_rect is not correct. Got = {bounding_rect}')
            else:
                self._bounding_rect = bounding_rect
        else:
            self._bounding_rect = None
        self._contour = contour
        self._area = None
        self._x_min = None
        self._y_min = None
        self._x_max = None
        self._y_max = None
        self._width = None
        self._center = None
        self._height = None
        self._rectangle = None

    @property
    def area(self) -> int:

        if self._area is None:
            self._area = self.width * self.height
        return self._area

    @property
    def bounding_rect(self) -> Tuple[int, int, int, int]:
        """
        :return: tuple with four elements: x, y, width, height
        """
        if self._bounding_rect is None:
            self._bounding_rect = cv2.boundingRect(self._contour)
        return self._bounding_rect

    @property
    def width(self) -> int:
        if self._width is None:
            self._width = self.bounding_rect[2]
        return self._width

    @property
    def height(self) -> int:
        if self._height is None:
            self._height = self.bounding_rect[3]
        return self._height

    @property
    def x_min(self) -> int:
        if self._x_min is None:
            self._x_min = self.bounding_rect[0]
        return self._x_min

    @property
    def y_min(self) -> int:
        if self._y_min is None:
            self._y_min = self.bounding_rect[1]
        return self._y_min

    @property
    def x_max(self) -> int:
        if self._x_max is None:
            self._x_max = self.x_min + self.width
        return self._x_max

    @property
    def y_max(self) -> int:
        if self._y_max is None:
            self._y_max = self.y_min + self.height
        return self._y_max

    @property
    def center(self) -> Tuple[int, int]:
        return int(self.x_min + (self.x_max - self.x_min) / 2), \
            int(self.y_min + (self.y_max - self.y_min) / 2)

    @property
    def rectangle(self) -> Tuple[int, int, int, int]:
        """
        :return: tuple with four elements: x_min, y_min, x_max, y_max
        """
        if self._rectangle is None:
            self._rectangle = self.x_min, self.y_min, self.x_max, self.y_max
        return self._rectangle
