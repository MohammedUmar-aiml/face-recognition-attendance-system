import numpy as np

def recognize_face(predictions, threshold=0.6):
    max_prob = np.max(predictions)
    if max_prob < threshold:
        return "Unknown"
    return np.argmax(predictions)
