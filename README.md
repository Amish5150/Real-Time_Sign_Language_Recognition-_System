# Real-Time Sign Language Recognition System

A computer vision and deep learning project that enables real-time recognition of hand signs using a webcam. The system detects hand gestures, preprocesses them into a standardized format, and classifies them using a trained neural network model.

Designed as an accessibility-focused AI application, this project demonstrates the integration of Computer Vision, Deep Learning, and Human-Computer Interaction to translate sign language gestures into machine-understandable outputs in real time.

---

# Overview

This project consists of two primary components:

### 1. Dataset Collection Pipeline

* Captures hand gesture images through a webcam
* Detects and isolates the hand region
* Preserves gesture proportions during preprocessing
* Places gestures on a standardized white background
* Generates datasets for model training

### 2. Real-Time Gesture Recognition

* Loads a trained deep learning model
* Performs live gesture classification
* Predicts sign language gestures in real time
* Displays recognized outputs directly from webcam input

---

# Supported Classes

The current model supports recognition of:

### Alphabet Signs

```
A  B  C  D  E  F  G
H  I  J  K  L  M  N
O  P  Q  R  S  T
U  V  W  X  Y  Z
```

### Numerical Signs

```
0 1 2 3 4 5 6 7 8 9
```

### Commands

```
OK
NO
```

### Total Classes

```
26 Alphabet Classes
10 Numerical Classes
2 Command Classes

Total: 38 Classes
```

---

# Features

* Real-time webcam-based gesture recognition
* Deep learning-powered classification
* Automatic hand detection and tracking
* Standardized image preprocessing pipeline
* Support for custom gesture expansion
* Lightweight local deployment
* Real-time inference with low latency
* Scalable dataset generation workflow

---

# Project Architecture

```text
Webcam Input
      │
      ▼
Hand Detection
      │
      ▼
Hand Cropping
      │
      ▼
Image Normalization
(300 × 300 Canvas)
      │
      ▼
Deep Learning Model
      │
      ▼
Gesture Classification
      │
      ▼
Predicted Output
```

---

# Dataset Collection

The dataset collection pipeline captures hand gestures through a webcam and automatically preprocesses them before storage.

Each captured image undergoes:

1. Hand Detection
2. Bounding Box Extraction
3. Aspect Ratio Preservation
4. Image Resizing
5. Center Alignment
6. White Canvas Normalization

This ensures that all training samples have consistent dimensions and minimal background noise, improving model performance and generalization.

### Dataset Structure

```text
Data/
│
├── A/
├── B/
├── C/
├── ...
├── Z/
│
├── 0/
├── 1/
├── 2/
├── ...
├── 9/
│
├── Ok/
└── No/
```

---

# Recommended Data Collection Strategy

The quality of the dataset directly influences model accuracy.

| Metric              | Recommendation |
| ------------------- | -------------- |
| Images per Class    | 1,000 – 2,000  |
| Total Classes       | 38             |
| Participants        | 10+            |
| Lighting Conditions | Multiple       |
| Background Types    | Multiple       |
| Camera Angles       | Multiple       |
| Hand Sizes          | Diverse        |

### Recommended Dataset Size

```text
38 Classes × 1500 Images

≈ 57,000 Images
```

A diverse dataset collected across different users, lighting conditions, and environments significantly improves model robustness and real-world performance.

---

# Model Training

The model was trained on custom sign language gesture data collected using the dataset generation pipeline.

### Training Recommendations

```text
70% Training Set
15% Validation Set
15% Testing Set
```

### Suggested Enhancements

* Data Augmentation
* Early Stopping
* Learning Rate Scheduling
* Cross Validation
* Class Balancing

---

# Real-Time Inference

The recognition system processes webcam frames continuously and performs gesture classification in real time.

### Inference Pipeline

```text
Webcam Frame
      │
      ▼
Hand Detection
      │
      ▼
Hand Crop
      │
      ▼
Normalization
      │
      ▼
Model Prediction
      │
      ▼
Gesture Output
```

The model predicts one of the 38 supported gesture classes and displays the result instantly.

---

# Technology Stack

## Computer Vision

* OpenCV
* CVZone
* MediaPipe Hand Tracking

## Machine Learning

* TensorFlow
* Keras
* Convolutional Neural Networks (CNN)

## Programming Language

* Python 3.x

---

# Installation

```bash
git clone https://github.com/Amish5150/Sign-Language-Recognition.git

cd Sign-Language-Recognition

pip install -r requirements.txt
```

### Required Libraries

```bash
pip install opencv-python
pip install cvzone
pip install mediapipe
pip install tensorflow
pip install numpy
```

---

# Running the Project

## Step 1 — Dataset Collection

```bash
python Data_Collection.py
```

Press:

```text
S → Save Current Gesture Image
```

Images are automatically processed and saved into their respective class folders.

---

## Step 2 — Train the Model

Train the model using the collected dataset.

Recommended split:

```text
70% Training
15% Validation
15% Testing
```

Save the trained model as:

```text
Model/
├── keras_model.h5
└── labels.txt
```

---

## Step 3 — Real-Time Prediction

```bash
python test.py
```

The application will:

* Detect the hand
* Extract the gesture region
* Normalize the image
* Run model inference
* Display the predicted sign

---

# Performance Expectations

Model performance depends on:

* Dataset size
* Dataset diversity
* Number of participants
* Lighting conditions
* Camera quality
* Gesture consistency

Typical performance ranges observed in CNN-based sign language recognition systems:

| Dataset Quality                        | Expected Accuracy |
| -------------------------------------- | ----------------- |
| Small Dataset (<300 images/class)      | 75–90%            |
| Medium Dataset (500–1000 images/class) | 90–96%            |
| Large Dataset (1000+ images/class)     | 96–99%+           |

With a sufficiently large and diverse dataset, this system can realistically achieve:

```text
92% – 98%
```

classification accuracy in controlled environments.

---

# Challenges Addressed

* Variations in hand size
* Different camera distances
* Lighting changes
* Background clutter
* Gesture scaling issues
* Real-time processing requirements
* Consistent image normalization

---

# Future Improvements

* Dynamic sign recognition using LSTMs
* Transformer-based gesture understanding
* Continuous sentence generation
* Text-to-Speech integration
* Mobile application deployment
* Multi-hand gesture recognition
* Multilingual sign language support
* Cloud-based inference APIs

---

# Applications

This project has applications across multiple domains:

* Accessibility Technology
* Sign Language Translation
* Human-Computer Interaction
* Robotics Control Systems
* Smart Healthcare
* Educational Platforms
* AR/VR Interfaces
* Gesture-Based User Interfaces

---

# Why This Project Matters

Over 70 million people worldwide rely on sign language as a primary means of communication. However, communication barriers still exist between sign language users and the broader population.

This project demonstrates how Artificial Intelligence and Computer Vision can be leveraged to create more inclusive technologies that improve accessibility, communication, and independence. Beyond sign language translation, the techniques developed here are directly applicable to real-world AI systems involving gesture recognition, assistive technology, robotics, and intelligent interfaces.

---

# Author

**Amish Mayank Ashar**

