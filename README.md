# A-deep-transfer-learning-framework-for-Pneumonia-Detection-Using-Chest-X-Rays-images

## Project Title
A Deep Transfer Learning Framework for Pneumonia Detection Using Chest X-Ray Images

---

## Abstract
This project focuses on developing an efficient deep learning system to classify chest X-ray images into three categories: Normal, Bacterial Pneumonia, and Viral Pneumonia. The system uses transfer learning techniques and evaluates multiple deep learning models. Among them, DenseNet121 achieved the highest accuracy and was selected for deployment.

The application is implemented as a web-based system to assist in early screening and academic research related to pneumonia detection.

---

## Dataset
The dataset used in this project is a refined version of the Chest X-Ray Pneumonia dataset available on Kaggle. The pneumonia class is further divided into bacterial and viral pneumonia categories.

Dataset Source:  
https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

---

## Models Used
The following transfer learning models were evaluated:

- VGG16  
- VGG19  
- ResNet50  
- ResNet101  
- DenseNet121 (Best Performing Model)

DenseNet121 achieved an accuracy of approximately **93%** and was selected for deployment.

---

## System Architecture
The system consists of the following components:

- Frontend: Web interface for image upload and result display  
- Backend: Flask server handling validation, preprocessing, and inference  
- Model: Trained DenseNet121 deep learning model  

---

## Workflow
1. User opens the web application.
2. User uploads a chest X-ray image.
3. Image is sent to the backend (Flask).
4. Input validation ensures only chest X-ray images are accepted.
5. Image preprocessing (resize, normalization) is performed.
6. The preprocessed image is passed to the trained DenseNet121 model.
7. The model predicts the class and confidence score.
8. Result is displayed to the user.

---

## Features
- Multi-class pneumonia classification
- Strict input validation (accepts only chest X-ray images)
- Confidence score for predictions
- Light and dark mode user interface
- Academic and research-oriented design

---

## Technologies Used
- Python
- Flask
- PyTorch
- Torchvision
- HTML, CSS, Bootstrap
- NumPy
- Pillow
ository-link>
cd Pneumonia-Detection-Website
