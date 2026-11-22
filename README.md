# Handwritten Digit Classifier – VisionAI

## Project Overview
This project identifies handwritten digits (0–9) using Machine Learning algorithms.

It compares different classification models and uses K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) to recognize handwritten digits.

In Week 3, an interactive web application was developed using Gradio, allowing users to draw or upload handwritten digit images and get predictions from the trained SVM model.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Gradio
- Pillow (PIL)
- Joblib
- Google Colab

## Dataset
1. Scikit-learn Digits Dataset (8×8 images)
2. MNIST Dataset (28×28 images)

## Machine Learning Models
The following four models were compared:

- Logistic Regression
- K-Nearest Neighbors (KNN)
- Support Vector Machine (SVM)
- Random Forest

## Model Accuracy Comparison

| Model | Accuracy |
|---|---:|
| Logistic Regression | 95.34% |
| KNN | 98.33% |
| SVM | 98.61% |
| Random Forest | 97.36% |

## Cross-Validation
Used 5-Fold Cross-Validation to compare the performance of four machine learning models.

## MNIST Results
- Dataset: MNIST
- Image Size: 28×28
- Algorithm: Support Vector Machine (SVM)
- Test Accuracy: 96.47%

## Model Evaluation
- Accuracy Comparison Graph
- Confusion Matrix
- Classification Report
- Misclassified Images Visualization

## Model Saving
The trained MNIST SVM model was saved using Joblib.

File: `digit_model.joblib`

## Week 3 – Gradio Web Application

Developed an interactive handwritten digit recognition web application using Gradio.

### Application Features
- Draw handwritten digits using an interactive canvas.
- Upload handwritten digit images.
- Recognize digits from 0 to 9.
- Display Top 3 predictions.
- Show prediction confidence scores.
- Interactive and user-friendly interface.

### Image Preprocessing
- Convert images into grayscale.
- Detect and invert the background when required.
- Crop unnecessary blank areas.
- Resize images to 28×28 pixels.
- Normalize pixel values.
- Flatten images into 784 features.

### Model Integration
Integrated the previously trained SVM model with the Gradio application using Joblib.

### Application Testing
Tested the application by drawing different handwritten digits.

The model successfully recognized most of the manually drawn digits. Some predictions were incorrect due to variations in handwriting and image preprocessing.

## Project Files
- `Handwritten_Digit_Classifier.ipynb` – Model training and evaluation.
- `app.py` – Gradio web application.
- `digit_model.joblib` – Saved trained SVM model.
- `requirements.txt` – Required Python libraries.
- `README.md` – Project documentation.

## How to Run the Application

1. Install the required libraries:

   `pip install -r requirements.txt`

2. Run the application:

   `python app.py`

3. Open the Gradio-generated link in your browser.

## Learning Outcomes
- Compared multiple Machine Learning algorithms.
- Learned model evaluation and cross-validation.
- Understood image preprocessing techniques.
- Integrated a trained ML model with Gradio.
- Developed an interactive digit recognition application.
- Learned to save and reuse trained ML models.

## Conclusion
The project successfully classifies handwritten digits using Machine Learning.

SVM achieved 98.61% accuracy on the original Scikit-learn Digits Dataset and 96.47% test accuracy on the MNIST Dataset.

In Week 3, the trained model was integrated into an interactive Gradio web application, making handwritten digit recognition easier and more accessible.

## Internship
Machine Learning Virtual Internship – SkillMe

**Project:** VisionAI – Local Computer Vision & Multi-Modal Intelligence Studio
