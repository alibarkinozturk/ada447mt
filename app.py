import gradio as gr
import pathlib
import os
from fastai.vision.all import *

# WindowsPath sorununu çöz
temp = pathlib.PosixPath
pathlib.WindowsPath = temp

learn = load_learner("./asl_sign_language_model.pkl")


THRESHOLD = 0.7

def predict(img):
    img = img.transpose(method=Image.FLIP_LEFT_RIGHT)
    pred, pred_idx, probs = learn.predict(img)
    confidence = probs[pred_idx].item() 
    if confidence >= THRESHOLD:
        return f"Prediction: {pred} (Confidence: {confidence:.4f})"
    else:
        return f"Not confident enough to make a prediction. (Confidence: {confidence:.4f})"

interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil", label="Upload image"),
    outputs="text",
    title="ASL Detection",
    description="To get the guessed letter, take a photo of yourself making one of the ASL letters with your webcam or upload a photo directly."
)

if __name__ == "__main__":
    interface.launch()