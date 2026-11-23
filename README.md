# VisionAI – Handwritten Digit Classifier

## Project Overview
VisionAI is a handwritten digit recognition application that identifies digits from 0 to 9 using Machine Learning and Deep Learning models.

This project compares two models:
- Support Vector Machine (SVM)
- Convolutional Neural Network (CNN)

The application provides an interactive interface using Gradio, allowing users to draw digits or upload handwritten digit images and receive predictions.

## Technologies Used
- Python
- NumPy
- Scikit-learn
- TensorFlow / Keras
- Gradio
- Pillow
- Joblib
- Matplotlib

## Dataset
The project uses the MNIST handwritten digit dataset.

- Training images: 60,000
- Testing images: 10,000
- Image dimensions: 28 × 28 pixels
- Classes: 0–9

## Models Implemented

### 1. Support Vector Machine (SVM)
- Implemented using Scikit-learn.
- Trained on the MNIST dataset.
- Used for handwritten digit classification.

### 2. Convolutional Neural Network (CNN)
- Implemented using TensorFlow and Keras.
- Contains two convolutional layers and two max-pooling layers.
- Uses ReLU and Softmax activation functions.
- Trained for 5 epochs.

## Model Performance

### MNIST Test Dataset Results

| Model | Test Accuracy |
|---|---:|
| SVM | 96.47% |
| CNN | 98.79% |

### Real-World Handwritten Testing

Both models were tested using the same 20 handwritten digit drawings.

| Model | Correct Predictions | Incorrect Predictions | Accuracy |
|---|---:|---:|---:|
| SVM | 18/20 | 2 | 90% |
| CNN | 18/20 | 2 | 90% |

## Application Features
- Interactive Gradio interface.
- Handwritten digit sketchpad.
- Image upload functionality.
- Model selection dropdown.
- Switch between SVM and CNN.
- Top 3 prediction results.
- Prediction scores.

## Project Structure

```text
Handwritten_Digit_Classifier/
│
├── app.py
├── digit_model.joblib
├── digit_cnn.keras
├── CNN_Training.ipynb
├── cnn_accuracy.png
├── cnn_loss.png
├── README.md
└── requirements.txt
```

## How to Run the Project

### Step 1: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Run the Application

```bash
python app.py
```

### Step 3: Open the Application

Open the Gradio local URL displayed in the terminal.

Draw a digit or upload an image, select SVM or CNN, and click Predict.

## Results and Conclusion

The CNN achieved 98.79% accuracy on the MNIST test dataset, outperforming the SVM model, which achieved 96.47%.

In real-world testing, both models achieved 90% accuracy across 20 handwritten drawings.

This project demonstrates the effectiveness of combining traditional Machine Learning and Deep Learning for handwritten digit recognition.

