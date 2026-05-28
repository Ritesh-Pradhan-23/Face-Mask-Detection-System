# 😷 Face Mask Detection System

An AI-powered computer vision application that detects whether a person in a given image is wearing a face mask or not. Built using Deep Learning and Convolutional Neural Networks (CNN), this tool provides a rapid, automated safety check mechanism.

## 📊 Dataset & Model Performance
* **Dataset**: Custom image dataset containing 3,833 total images (3,066 used for training, 767 for testing).
* **Classes**: No Mask (Class 0) and Mask (Class 1).
* **Architecture**: Utilizes **VGG16** (pre-trained on ImageNet) for transfer learning. The custom classification head includes: `Flatten() → Dense(128, ReLU) → Dropout(0.4) → Dense(2, Softmax)`.
* **Performance**: Achieved an overall **91% testing accuracy** with an F1-score of 0.92 for mask detection. 

## 🛠️ Tech Stack
* **Deep Learning**: TensorFlow & Keras
* **Computer Vision**: OpenCV (`cv2`) for image reading, resizing to 200x200, and standardizing inputs.
* **Frontend UI**: Streamlit for the interactive web deployment.
* **Data Processing & Metrics**: NumPy, Pandas, and Scikit-learn (Confusion Matrix & Classification Report).

## 🚀 Web App Features
The project includes a live Streamlit web application where users can upload an image (JPG, JPEG, PNG).
* **Automated Preprocessing**: Images are resized to 200x200 and normalized.
* **Channel Consistency (RGB to BGR)**: The app explicitly converts PIL RGB images to BGR format using NumPy slicing (`img[:, :, ::-1]`) to match the `cv2.imread()` pipeline used during training, ensuring prediction accuracy.
* **Probability Output**: Displays the raw prediction probabilities alongside visual success/error alerts based on the model's highest confidence class.

## ⚙️ How to Run Locally

1. **Clone the repository**:
   ```bash
   git clone [https://github.com/YourUsername/Face-Mask-Detection.git](https://github.com/YourUsername/Face-Mask-Detection.git)
   cd Face-Mask-Detection
