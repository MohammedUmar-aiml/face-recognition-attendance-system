import cv2
import numpy as np

def preprocess_image(img, target_size=(128, 128)):
    """
    Resize and normalize an input image.
    """
    img = cv2.resize(img, target_size)
    img = img.astype("float32") / 255.0
    return img


if __name__ == "__main__":
    print("Preprocessing function defined")

