# Face Recognition Attendance System

A CNN-based face recognition system developed to automate attendance marking.

---

## Overview
This project demonstrates a basic face recognition pipeline using deep learning.
The goal is to understand preprocessing, model training, and limitations such as
false positive matches when trained on small datasets.

---

## Pipeline
Image Input → Preprocessing → CNN Model → Identity Prediction → Attendance Logic

---

## Project Structure

face-recognition-attendance-system/
│
├── src/
│   ├── preprocess.py        # Image preprocessing functions
│   ├── train_model.py       # CNN model definition and training logic
│   └── recognize.py         # Inference / recognition logic
│
├── notebooks/
│   └── training.ipynb       # Step-by-step training walkthrough
│
├── requirements.txt         # Pinned dependency versions
├── README.md
├── .gitignore
└── .gitattributes

---

## Dataset
The model is currently trained and tested on a small number of individuals.
The code is structured so that it can scale to larger datasets by adding more
training images.

---

## Challenges Observed
- False positive matches due to limited training data
- Sensitivity to lighting and pose variations
- Generalization issues with small datasets

---

## Future Improvements
- Increase dataset size
- Use embedding-based face recognition (FaceNet-style)
- Improve matching threshold logic
- Add real-time camera support
