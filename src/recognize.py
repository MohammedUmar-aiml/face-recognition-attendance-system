import tensorflow as tf
import cv2
import numpy as np
from preprocess import preprocess_image

def recognize_face(model_path, img_path):
    model = tf.keras.models.load_model(model_path)
    img = cv2.imread(img_path)
    img = preprocess_image(img)
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    return prediction

print("Recognition module ready")
