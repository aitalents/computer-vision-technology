from abc import ABC, abstractmethod

import io
from PIL import Image, UnidentifiedImageError
import numpy as np
import requests


class ImageDownloadError(Exception):
    def __init__(self, message: str = 'Error while downloading image'):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class ImageAsArrayError(Exception):

    def __init__(self, message: str = 'Error while converting image to array'):
        # Call the base class constructor with the parameters it needs
        super().__init__(message)


class ImageLoader(ABC):
    @abstractmethod
    def __init__(self):
        pass

    @abstractmethod
    def load_image(self, url: str):
        raise NotImplemented

    def image_as_array(self, image_bytes: bytes) -> np.ndarray:
        try:
            shelf_image = np.array(Image.open(io.BytesIO(image_bytes)))
        except Exception:
            raise ImageAsArrayError
        return shelf_image


class RequestImageLoader(ImageLoader):
    def __init__(self):
        super().__init__()
        pass

    def load_image(self, url: str) -> bytes:
        r = requests.get(url)
        if r.ok:
            return r.content
        else:
            raise ImageDownloadError(
                'Failed to download: %s %s\n%s' % (r.status_code, r.reason, r.url)
            )


class LocalImageLoader(ImageLoader):
    def __init__(self):
        super().__init__()
        pass

    def load_image(self, url: str) -> bytes:
        with open(url, 'rb') as f:
            file = f.read()
        return file