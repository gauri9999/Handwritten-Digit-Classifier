# Handwritten Digit Classifier

## Project Overview
This project identifies handwritten digits (0–9) using Machine Learning algorithms. It compares different classification models and uses K-Nearest Neighbors (KNN) and Support Vector Machine (SVM) to recognize handwritten digits.

## Technologies Used
- Python
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
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
- SVM Test Accuracy: 95.7%

## Model Evaluation
- Accuracy Comparison Graph
- Confusion Matrix
- Classification Report
- Misclassified Images Visualization

## Model Saving
The trained MNIST model was saved using Joblib.

File: `digit_model.joblib`

## Project Features
- Handwritten digit recognition (0–9)
- Comparison of multiple ML algorithms
- Model evaluation using accuracy
- Confusion matrix visualization
- Prediction of handwritten digits
- Saving and loading trained models

## Conclusion
The project successfully classifies handwritten digits using Machine Learning. SVM achieved 98.61% accuracy on the original Digits dataset and 95.7% test accuracy on the MNIST dataset.

