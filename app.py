import gradio as gr # type: ignore
import numpy as np
import joblib

from PIL import Image, ImageOps


# =====================================
# 1. LOAD TRAINED MODEL
# =====================================

MODEL_PATH = "digit_model.joblib"

model = joblib.load(MODEL_PATH)

print("VisionAI model loaded successfully!")


# =====================================
# 2. CONVERT INPUT INTO PIL IMAGE
# =====================================

def convert_to_pil(image):

    if image is None:
        return None

    # Handle Gradio Sketchpad dictionary
    if isinstance(image, dict):

        image = image.get("composite")

        if image is None:
            raise ValueError("Please draw a digit first.")

    # Convert NumPy array into PIL image
    if isinstance(image, np.ndarray):

        if image.dtype != np.uint8:
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

    if image is None:
        raise ValueError("Please draw or upload an image.")

    # Convert to grayscale
    image = ImageOps.grayscale(image)

    # Invert image if background is white
    if np.array(image).mean() > 127:
        image = ImageOps.invert(image)

    # Find digit boundaries
    image_array = np.array(image)

    if image_array.max() == 0:
        raise ValueError("No digit detected. Please draw clearly.")

    # Crop around the digit
    bbox = image.getbbox()

    if bbox:
        image = image.crop(bbox)

    # Resize image to MNIST size
    image = ImageOps.pad(
        image,
        (20, 20),
        color=0
    )

    # Add 4-pixel padding
    image = ImageOps.expand(
        image,
        border=4,
        fill=0
    )

    # Ensure image is 28x28
    image = image.resize((28, 28))

    # Normalize pixel values
    image_array = np.array(image).astype("float32") / 255.0

    # Flatten image into 784 features
    image_array = image_array.reshape(1, 784)

    return image_array


# =====================================
# 4. PREDICT DIGIT
# =====================================

def predict_digit(image):

    try:

        # Preprocess input image
        processed = preprocess_image(image)

        # Predict digit
        prediction = model.predict(processed)[0]

        # Get SVM decision scores
        scores = model.decision_function(processed)

        scores = np.asarray(scores)

        # Convert decision scores into relative scores
        if scores.ndim == 1:
            scores = scores.reshape(1, -1)

        scores = scores[0]

        # Softmax conversion
        exp_scores = np.exp(scores - np.max(scores))

        probabilities = exp_scores / np.sum(exp_scores)

        # Get top 3 predictions
        top_indices = np.argsort(probabilities)[::-1][:3]

        results = {
            str(model.classes_[i]): float(probabilities[i])
            for i in top_indices
        }

        confidence = probabilities[
            np.where(model.classes_ == prediction)[0][0]
        ] * 100

        return results, (
            f"Predicted Digit: {prediction}\n"
            f"Relative Score: {confidence:.2f}%"
        )

    except Exception as error:

        print("ERROR DETAILS:", repr(error))

        return {}, f"Error: {str(error)}"


# =====================================
# 5. GRADIO INTERFACE
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
        Our Machine Learning model will predict the digit.
        """
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
                lines=3
            )

    # Prediction button
    predict_button.click(
        fn=predict_digit,
        inputs=[sketch],
        outputs=[output, result_text]
    )

    upload.change(
        fn=predict_digit,
        inputs=[upload],
        outputs=[output, result_text]
    )


# =====================================
# 6. LAUNCH APPLICATION
# =====================================

if __name__ == "__main__":

    print("Starting VisionAI application...")

    demo.launch(share=True)
