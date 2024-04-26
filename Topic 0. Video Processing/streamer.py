from typing import Union
from pathlib import Path
from abc import abstractmethod, ABC

import cv2


class VideoIterator(ABC):

    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def _generator_fn(self):
        yield None

    def __iter__(self):
        return self._generator_fn()


class FrameVideoIterator(VideoIterator):

    def __init__(self, video_source: Union[Path, str], frames_delay: int=150):
        """
        Args:
            video_source (Any[Path, str]): Source file path or rtsp url
        """
        self.source = video_source
        self.frames_delay = frames_delay
        self.shape = None

    def _videofile_frame_iterator(self):
        capture = cv2.VideoCapture(str(self.source))
        self.counter = -1
        delay_counter = 0
        while capture.isOpened():
            ret,frame = capture.read()
            if not ret:
                self.counter = 0
                break
            else:
                self.counter += 1
                yield self.counter, frame[:,:,::-1]

    def _generator_fn(self):
        video_gen = self._videofile_frame_iterator()
        for i, frame in video_gen:
            yield i, frame

