
import gradio as gr
import numpy as np
import joblib

from PIL import Image, ImageOps

# Load the trained model only once
MODEL_PATH = "/content/drive/MyDrive/SkillMe_Projects /digit_model.joblib"

model = joblib.load(MODEL_PATH)

print("VisionAI model loaded successfully!")


# Convert input into a PIL image
def convert_to_pil(image):

    if image is None:
        return None

    # Handle Gradio Sketchpad dictionary
    if isinstance(image, dict):
        image = image.get("composite")

    if image is None:
        return None

    if isinstance(image, np.ndarray):
        if image.size == 0:
            return None

        if image.dtype != np.uint8:
            image = np.nan_to_num(image)
            if image.max() <= 1:
                image = image * 255
            image = np.clip(image, 0, 255).astype(np.uint8)

        image = Image.fromarray(image)

    return image


# Preprocess image
def preprocess_image(image):

    image = image.convert("RGB")
    image = ImageOps.grayscale(image)

    # Invert white background into black
    arr = np.array(image)

    if arr.mean() > 127:
        image = ImageOps.invert(image)

    arr = np.array(image)

    # Detect empty canvas
    if arr.max() < 20:
        raise ValueError("Please draw a digit or upload a valid image.")

    # Resize to MNIST dimensions
    image = image.resize((28, 28))

    arr = np.array(image).astype("float32")

    # Normalize pixel values
    arr = arr / 255.0

    # Flatten into 784 features
    return arr.reshape(1, 784)


# Predict handwritten digit
def predict_digit(sketch, uploaded_image):

    try:

        # Uploaded image gets priority
        image = convert_to_pil(uploaded_image)

        # Otherwise use drawing canvas
        if image is None:
            image = convert_to_pil(sketch)

        if image is None:
            return {}, "Please draw or upload a digit."

        processed = preprocess_image(image)

        # Get probabilities
        probabilities = model.predict_proba(processed)[0]

        # Get top 3 predictions
        top_3 = np.argsort(probabilities)[::-1][:3]

        results = {
            str(int(digit)): float(probabilities[digit])
            for digit in top_3
        }

        best_digit = int(top_3[0])
        confidence = probabilities[best_digit] * 100

        message = (
            f"Prediction successful! "
            f"Digit: {best_digit} | "
            f"Confidence: {confidence:.2f}%"
        )

        return results, message

    except ValueError as error:
        return {}, str(error)

    except Exception as error:
        print("Error:", error)
        return {}, "Unable to process this image. Please try another image."


# Build Gradio application
with gr.Blocks(title="VisionAI - Digit Recognition") as demo:

    gr.Markdown(
        """
        # VisionAI - Handwritten Digit Recognition

        Draw a digit from 0 to 9 or upload a handwritten digit image.
        Our machine learning model will predict the result.
        """
    )

    with gr.Row():

        with gr.Column():
            sketch = gr.Sketchpad(
                label="Draw Your Digit",
                type="numpy",
                height=300,
                width=300
            )

            upload = gr.Image(
                type="pil",
                label="Upload Handwritten Digit"
            )

            predict_button = gr.Button(
                "Predict Digit",
                variant="primary"
            )

        with gr.Column():

            output = gr.Label(
                num_top_classes=3,
                label="Top 3 Predictions"
            )

            status = gr.Textbox(
                label="Prediction Status",
                interactive=False
            )

    predict_button.click(
        fn=predict_digit,
        inputs=[sketch, upload],
        outputs=[output, status]
    )

    gr.Markdown(
        "### Technology: Python | SVM | MNIST | Gradio"
    )


if __name__ == "__main__":

    demo.launch(share=True)
