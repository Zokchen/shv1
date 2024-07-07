import gradio as gr
from predict import predict_flower

def classify_image(image):
    predicted_class, confidence = predict_flower(image)
    return f"Predicted class: {predicted_class}\nConfidence: {confidence:.2f}"

iface = gr.Interface(
    fn=classify_image,
    inputs=gr.Image(type="filepath"),
    outputs="text",
    title="Image Classifier",
    description="Upload an image to classify it.",
)

iface.launch()