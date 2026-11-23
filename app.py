import gradio as gr
import numpy as np
import joblib
import tensorflow as tf

from PIL import Image, ImageOps
from pathlib import Path


# =====================================
# 1. LOAD BOTH TRAINED MODELS
# =====================================

BASE_DIR = Path(__file__).resolve().parent

SVM_PATH = BASE_DIR / "digit_model.joblib"
CNN_PATH = BASE_DIR / "digit_cnn.keras"

svm_model = joblib.load(SVM_PATH)
cnn_model = tf.keras.models.load_model(CNN_PATH)

print("SVM model loaded successfully!")
print("CNN model loaded successfully!")
print("VisionAI models are ready!")


# =====================================
# 2. CONVERT INPUT INTO PIL IMAGE
# =====================================

def convert_to_pil(image):

    if image is None:
        raise ValueError("Please draw or upload a digit first.")

    # Handle Gradio Sketchpad dictionary
    if isinstance(image, dict):

        image = image.get("composite")

        if image is None:
            raise ValueError("Please draw a digit first.")

    # Convert NumPy array into PIL image
    if isinstance(image, np.ndarray):

        if np.issubdtype(image.dtype, np.floating):
            if image.size > 0 and image.max() <= 1.0:
                image = image * 255

        image = np.clip(image, 0, 255).astype(np.uint8)
        image = Image.fromarray(image)

    if not isinstance(image, Image.Image):
        raise ValueError("Invalid image format.")

    return image


# =====================================
# 3. IMAGE PREPROCESSING
# =====================================

def preprocess_image(image):

    image = convert_to_pil(image)

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Invert if background is white
    if np.array(image).mean() > 127:
        image = ImageOps.invert(image)

    image_array = np.array(image)

    # Check whether a digit exists
    if image_array.max() == 0:
        raise ValueError("No digit detected. Please draw clearly.")

    # Crop around digit
    bbox = image.getbbox()

    if bbox:
        image = image.crop(bbox)

    # Resize while maintaining proportions
    image = ImageOps.pad(
        image,
        (20, 20),
        color=0
    )

    # Add padding
    image = ImageOps.expand(
        image,
        border=4,
        fill=0
    )

    # Resize to MNIST dimensions
    image = image.resize((28, 28))

    # Normalize pixels
    image_array = np.array(image).astype("float32") / 255.0

    return image_array


# =====================================
# 4. SVM PREDICTION
# =====================================

def predict_svm(processed):

    # Flatten image for SVM
    svm_input = processed.reshape(1, 784)

    prediction = svm_model.predict(svm_input)[0]

    # Get decision scores
    scores = svm_model.decision_function(svm_input)

    scores = np.asarray(scores).reshape(-1)

    # Convert scores into relative ranking scores
    exp_scores = np.exp(scores - np.max(scores))
    probabilities = exp_scores / np.sum(exp_scores)

    classes = svm_model.classes_

    top_indices = np.argsort(probabilities)[::-1][:3]

    results = {
        str(classes[i]): float(probabilities[i])
        for i in top_indices
    }

    predicted_index = np.where(classes == prediction)[0][0]

    confidence = probabilities[predicted_index] * 100

    message = (
        f"Model: SVM\n"
        f"Predicted Digit: {prediction}\n"
        f"Relative Score: {confidence:.2f}%"
    )

    return results, message


# =====================================
# 5. CNN PREDICTION
# =====================================

def predict_cnn(processed):

    # CNN input shape: (1, 28, 28, 1)
    cnn_input = processed.reshape(1, 28, 28, 1)

    probabilities = cnn_model.predict(
        cnn_input,
        verbose=0
    )[0]

    # Get predicted digit
    prediction = int(np.argmax(probabilities))

    # Get top 3 predictions
    top_indices = np.argsort(probabilities)[::-1][:3]

    results = {
        str(int(i)): float(probabilities[i])
        for i in top_indices
    }

    confidence = probabilities[prediction] * 100

    message = (
        f"Model: CNN\n"
        f"Predicted Digit: {prediction}\n"
        f"Confidence: {confidence:.2f}%"
    )

    return results, message


# =====================================
# 6. MAIN PREDICTION FUNCTION
# =====================================

def predict_digit(image, model_choice):

    try:

        processed = preprocess_image(image)

        if model_choice == "SVM (Scikit-learn)":

            results, message = predict_svm(processed)

        elif model_choice == "CNN (TensorFlow)":

            results, message = predict_cnn(processed)

        else:

            raise ValueError("Please select a valid model.")

        return results, message

    except Exception as error:

        print("ERROR DETAILS:", repr(error))

        return {}, f"Prediction error: {str(error)}"


# =====================================
# 7. GRADIO INTERFACE
# =====================================

with gr.Blocks(
    theme=gr.themes.Soft(),
    title="VisionAI - Handwritten Digit Classifier"
) as demo:

    gr.Markdown(
        """
        # 🧠 VisionAI
        ### Handwritten Digit Recognition System

        Draw a digit (0–9) or upload an image.

        Compare predictions using Machine Learning
        and Deep Learning models.
        """
    )

    model_selector = gr.Dropdown(
        choices=[
            "SVM (Scikit-learn)",
            "CNN (TensorFlow)"
        ],
        value="SVM (Scikit-learn)",
        label="Select Prediction Model"
    )

    with gr.Row():

        with gr.Column():

            sketch = gr.Sketchpad(
                label="Draw Your Digit Here",
                type="numpy",
                height=300,
                width=300
            )

            upload = gr.Image(
                label="Or Upload an Image",
                type="pil"
            )

            predict_button = gr.Button(
                "🔍 Predict Digit",
                variant="primary"
            )

            clear_button = gr.ClearButton(
                components=[sketch, upload],
                value="🗑️ Clear"
            )

        with gr.Column():

            output = gr.Label(
                label="Top 3 Predictions",
                num_top_classes=3
            )

            result_text = gr.Textbox(
                label="Prediction Result",
                lines=4
            )

    # Predict using sketchpad
    predict_button.click(
        fn=predict_digit,
        inputs=[sketch, model_selector],
        outputs=[output, result_text]
    )

    # Predict automatically after uploading image
    upload.change(
        fn=predict_digit,
        inputs=[upload, model_selector],
        outputs=[output, result_text]
    )


# =====================================
# 8. LAUNCH APPLICATION
# =====================================

if __name__ == "__main__":

    print("Starting VisionAI application...")

    demo.launch(share=True)
