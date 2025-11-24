# Roadside Sign Recognition System

This project implements a **Deep Learning--based Roadside Sign
Recognition System**.\
The Jupyter notebook (`new_analysis.ipynb`) contains the model
development pipeline, including data preprocessing, CNN training,
evaluation, and saving the final model.\
The `app.py` file provides a **Streamlit web application** where users
can upload a traffic sign image, and the trained model predicts its
category.

## Live Demo

You can view and interact with the deployed Streamlit app here:

**[Live App Link](https://roadside-sign-prediction-naqshabjaved.streamlit.app/)** 
---

## 📌 Project Overview

-   Built using a **Convolutional Neural Network (CNN)** trained on
    traffic sign datasets.\
-   Supports **43 traffic sign classes** such as speed limits, warning
    signs, prohibitory signs, and mandatory direction signs.\
-   The Streamlit app allows users to upload an image and receive
    predictions with confidence scores.\
-   The model is loaded from `notebooks/traffic_classifier.h5` and is
    automatically cached for faster performance.\
-   Handles grayscale and RGBA images by converting them to RGB and
    resizing them to `64×64`.

## 📁 Project Structure

    ├── app.py                     # Streamlit UI for prediction
    ├── notebooks/
    │   ├── traffic_classifier.h5  # Saved trained model
    │   └── new_analysis.ipynb     # Model training & analysis
    └── README.md




##  How the App Works

1.  Upload an image (`.jpg`, `.jpeg`, `.png`)\
2.  Image is resized to 64×64 and normalized\
3.  CNN model predicts the class\
4.  Displays:
    -   Predicted label\
    -   Confidence percentage\
    -   Uploaded image preview

##  Technologies Used

-   Python\
-   TensorFlow / Keras\
-   Streamlit\
-   NumPy\
-   Pillow (PIL)

##  Future Enhancements

-   Add webcam-based detection\
-   Deploy the model on cloud (AWS/GCP/Render)\
-   Improve accuracy with more training data or augmentation

## How to Run This Project Locally

### Prerequisites
- Python 3.10+
- `pip` (Python package installer)

### 1. Clone the Repository
```bash
git clone [https://github.com/naqshabjaved/Roadside-sign-recognition.git](https://github.com/naqshabjaved/Roadside-sign-recognition.git)
cd Roadside-sign-recognition