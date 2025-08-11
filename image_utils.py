import cv2
import numpy as np
from pathlib import Path
from io import BytesIO

def load_cv_image(image_input):
    """Load image into an OpenCV numpy array."""
    if isinstance(image_input, (str, Path)):
        img = cv2.imread(str(image_input), cv2.IMREAD_UNCHANGED)
    elif isinstance(image_input, (bytes, bytearray)):
        img = cv2.imdecode(np.frombuffer(image_input, np.uint8), cv2.IMREAD_UNCHANGED)
    elif isinstance(image_input, BytesIO):
        img = cv2.imdecode(np.frombuffer(image_input.getvalue(), np.uint8), cv2.IMREAD_UNCHANGED)
    else:
        raise ValueError("Unsupported input type. Use path, bytes, bytearray or BytesIO.")
    return img

def cv2_to_rgb_for_model(cv2_img):
    """Convert cv2 image to RGB (no alpha) for model inputs."""
    if cv2_img is None:
        return None
    if cv2_img.ndim == 2:
        return cv2.cvtColor(cv2_img, cv2.COLOR_GRAY2RGB)
    if cv2_img.shape[2] == 4:
        return cv2.cvtColor(cv2_img, cv2.COLOR_BGRA2RGB)
    return cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)
