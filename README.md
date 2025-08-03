# Real-Time Posture Classification

A simple demo that captures webcam video, uses MediaPipe to detect pose landmarks, computes key tilt angles (neck, shoulders, eyes, mouth), and classifies your posture with a Random Forest Classifier model. 

NOTE: The dataset of pose samples is very small, so the classifier is almost certainly overfitted. Add your own data using collecting_landmark_data.py!

---

## 🚀 Features

- **Live video capture** via OpenCV  
- **Pose detection** with MediaPipe’s PoseLandmarker  
- **Angle calculations** for neck, shoulder, eye and mouth tilt  
- **Posture classification** using a scikit-learn model  
- **On-screen annotations** of landmarks and predicted class

---

## 🛠️ Prerequisites

- **Python 3.7+**  
- A working **webcam**

---

## ⚙️ Installation

1. **Clone the repo**  
   ```bash
   git clone <url>
   cd pose-estimation
   pip install -r requirements.txt
