# Emotion-Detection-System-using-OpenCV-and-CNN
Machine Learning project on Emotion Detection System
🎯 Human Emotion and Gesture Recognition System

Developed by Amita Sharma

A deep learning–powered project integrating Facial Emotion Recognition and Finger Count Detection using TensorFlow, Keras, and OpenCV.
This system enables computers to understand human emotions from facial expressions and count fingers using real-time video input — bridging human-computer interaction through vision-based AI.


📘 Overview

This repository combines two intelligent computer vision modules:

Emotion Recognition:
Detects human faces and classifies their emotional state (e.g., Happy, Sad, Angry, Fearful, Neutral, Disgust, Surprise) using a CNN trained on the FER-2013 dataset.

Finger Count Detection:
Utilizes contour and convex hull analysis to count the number of fingers shown to the webcam in real time.

Together, these modules demonstrate the potential of AI in emotion-aware systems, gesture-based interfaces, and assistive technologies.


🧠 Project Architecture
Emotion Recognition

Dataset: FER-2013

Model: Convolutional Neural Network (CNN)

Input Size: 48×48 grayscale images

Framework: TensorFlow/Keras

Classes: 7 Emotion Categories

Training: 10 epochs with data augmentation

Optimizer: Adam

Loss Function: Categorical Cross-Entropy

Finger Count Detection

Libraries Used: OpenCV, NumPy

Technique:

ROI (Region of Interest) extraction

Skin color thresholding (HSV filtering)

Contour and Convex Hull detection

Convexity defects analysis to count raised fingers

⚙ Tech Stack
Component	Technology
Language	Python 3
Deep Learning	TensorFlow, Keras
Computer Vision	OpenCV
Data Handling	NumPy
Visualization	Matplotlib
Dataset	FER-2013

📂 Repository Structure
📦 Human-Emotion-and-Gesture-Recognition
│
├── Emotion_Recognition.ipynb       # CNN-based emotion detection
├── Fingercount.ipynb               # Finger count detection module
├── fer2013/                        # Dataset directory (train/test split)
│   ├── train/
│   └── test/
└── README.md


🧩 Features

✅ Real-time face detection and emotion classification
✅ Finger count detection using webcam feed
✅ Lightweight and modular architecture
✅ Compatible with standard webcams
✅ Eager execution and debugging support
✅ Automatically saves and loads trained models


🚀 How to Run
1️⃣ Clone the Repository

2️⃣ Install Dependencies
pip install tensorflow opencv-python numpy matplotlib

3️⃣ Run Emotion Recognition
jupyter notebook Emotion_Recognition.ipynb

4️⃣ Run Finger Count Detection
jupyter notebook Fingercount.ipynb


📸 Emotion Recognition Demo
📸 Finger Count Detection Demo


📊 Model Performance
Metric	Value
Training Accuracy	~90%
Validation Accuracy	~85%
Optimizer	Adam
Loss Function	Categorical Cross-Entropy
Epochs	10

(You can replace these values with your actual results.)


🧾 Applications

Human-Computer Interaction

Emotion-Aware Systems

Virtual Assistants

Gaming Interfaces

Sign Language Recognition (extended use)

Mental Health Monitoring


🔍 Future Enhancements

Integrate both modules into a single unified GUI

Improve model accuracy using deeper CNN architectures

Add real-time emotion analytics dashboard

Integrate gesture and emotion input for hybrid control systems


🧑‍💻 Author

Amita Sharma
🎓 Major Project | Academic Submission
📍 India



⭐ Acknowledgments

FER-2013 Dataset — for providing labeled facial expression data

TensorFlow/Keras — for deep learning framework support

OpenCV — for real-time computer vision and gesture detection

Matplotlib & NumPy — for visualization and numerical analysis
