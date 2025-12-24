"""
CNN training script for face recognition attendance system.
Demonstrates model architecture, compilation, and training flow.
"""
import tensorflow as tf
from tensorflow.keras import layers, models


def build_cnn(input_shape=(128, 128, 3), num_classes=3):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax")
    ])
    return model


def compile_model(model):
    model.compile(
        optimizer="adam",
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


if __name__ == "__main__":
    model = build_cnn()
    model = compile_model(model)

    print("CNN model built and compiled successfully")
if __name__ == "__main__":
    print("Training script ready. Dataset integration required to start training.")
