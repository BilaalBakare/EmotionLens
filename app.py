import gradio as gr
import torch
from src.model import Emolens
import cv2
from torchvision import transforms
import torch.nn.functional as F
from collections import deque

# Load model once at startup
Emodel = Emolens()
state_dict = torch.load('models/affectnet-weighted_loss.pth', map_location='cpu', weights_only=True)
Emodel.load_state_dict(state_dict)
Emodel.eval()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
temperature = 2.0
prediction_buffer = deque(maxlen=7)


def detect_emotion(frame):
    frame = frame.copy()
    cv2.imwrite("debug_frame.png", frame)

    if frame is None:
        return frame

    # Gradio gives RGB, OpenCV cascade wants grayscale
    gray_image = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    gray_image = clahe.apply(gray_image)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    for (x, y, w, h) in faces:
        if w < 100 or h < 100:
            continue  # skip detections that are too small to be a real face

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_crop = frame[y:y + h, x:x + w]

        transformed_tensor = transform(face_crop).unsqueeze(0)

        with torch.inference_mode():
            predictions = Emodel(transformed_tensor)
        probabilities = F.softmax(predictions / temperature, dim=1)

        prediction_buffer.append(probabilities)

        if len(prediction_buffer) == 7:
            avg_probs = torch.mean(torch.stack(list(prediction_buffer), dim=0), dim=0)
            confidence, predicted_class = torch.max(avg_probs, dim=1)
            confidence = confidence.item()
            prediction = predicted_class.item()

            # print(f"Prediction: {prediction}, Confidence: {confidence}")

            label = f"{emotions[prediction]} ({confidence:.2f})" if confidence >= 0.45 else "Detecting"

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

    return frame


# ---- Custom theme and styling ----
theme = gr.themes.Soft(
    primary_hue="emerald",
    neutral_hue="slate",
    font=[gr.themes.GoogleFont("Inter"), "sans-serif"]
)

custom_css = """
#title { text-align: center; margin-bottom: 0; }
#subtitle { text-align: center; color: #94a3b8; margin-top: 0; font-size: 1.05rem; }
#author-note { text-align: center; color: #64748b; font-size: 0.95rem; margin-top: -8px; }
#tip { text-align: center; color: #64748b; font-size: 0.9rem; margin-top: 4px; }
#links { text-align: center; margin-top: 12px; }
footer { visibility: hidden; }
"""

with gr.Blocks(theme=theme, css=custom_css, title="EmotionLens") as demo:
    gr.Markdown("# EmotionLens", elem_id="title")
    gr.Markdown(
        "Recognizing facial emotions from a live video feed using a fine-tuned deep learning model.",
        elem_id="subtitle"
    )
    gr.Markdown(
        "A computer vision project developed by **Bilaal Bakare** to explore real-time emotion recognition.",
        elem_id="author-note"
    )

    gr.Markdown(
        "For best results, please ensure your face is clearly visible and well lit.",
        elem_id="tip"
    )

    cam = gr.Image(sources=["webcam"], streaming=True, label=None, show_label=False)
    cam.stream(fn=detect_emotion, inputs=cam, outputs=cam)

    gr.Markdown(
        "[LinkedIn](https://www.linkedin.com/in/bilaal-bakare) &nbsp;&nbsp;|&nbsp;&nbsp; "
        "[GitHub](https://github.com/BilaalBakare)",
        elem_id="links"
    )

demo.launch()