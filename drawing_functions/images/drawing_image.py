import logging
import os

import numpy as np
from PIL import Image


class DrawingImage:
    """Metadata for an image, including its shape and filename."""

    image_name: str
    shape: tuple[int, int, int]  # (height, width, channels)
    aspect_ratio: float
    image: Image.Image

    def __init__(self, image_name: str):
        image_path = os.path.join(
            os.path.join(os.path.dirname(__file__)), f"{image_name}.png"
        )
        logging.debug(f"Attempting to load image: {image_path}")

        if not os.path.exists(image_path):
            logging.error(f"Image file does not exist: {image_path}")
            raise FileNotFoundError(f"Image file not found: {image_path}")

        try:
            image = Image.open(image_path)
            logging.debug(f"Successfully loaded image: {image_path}")
            # Store metadata
            self.image = image
        except Exception as e:
            logging.error(f"Failed to load image {image_path}: {str(e)}")
            raise

        self.filename = image_path
        img_ndarray = np.array(image)
        self.shape = img_ndarray.shape
        self.aspect_ratio = img_ndarray.shape[0] / img_ndarray.shape[1]
