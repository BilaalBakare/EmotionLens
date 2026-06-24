import torch
from model import Emolens
import cv2
from torchvision import datasets, transforms
import torch.nn.functional as F
from collections import deque

prediction_buffer = deque(maxlen=7)

model_path = '/home/bbo/Documents/Code/projects/Emotionlens/models/affectnet-weighted_loss.pth'

Emodel = Emolens()
state_dict = torch.load(model_path, map_location='cpu', weights_only=True)
Emodel.load_state_dict(state_dict)
Emodel.eval()

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open the webcam.")
    exit()

transform = transforms.Compose([
        transforms.ToPILImage(),   
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
emotions = ['anger', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
temprature = 2.0
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))

while True:
    ret, frame = cap.read()
    print(f"Local webcam frame shape: {frame.shape}")

    if not ret:
        print("Error: Can't receive frame. Exiting...")
        break

    gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_image = clahe.apply(gray_image)

    faces = face_cascade.detectMultiScale(gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face_crop = frame[y:y+h, x:x+w]

        face_rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
        transformed_tensor = transform(face_rgb)
        transformed_tensor = transformed_tensor.unsqueeze(0) 

        with torch.inference_mode():
            predictions = Emodel(transformed_tensor)
        probabilities = F.softmax(predictions / temprature, dim=1)

        prediction_buffer.append(probabilities)

        if len(prediction_buffer) == 7: 
            avg_probs = torch.mean(torch.stack(list(prediction_buffer), dim=0), dim=0)
            confidences, predicted_classes = torch.max(avg_probs, dim=1)
            prediction = predicted_classes[0]
            confidence = confidences[0]

            if confidence >= 0.45:
                label = f"{emotions[prediction]} ({confidence:.2f})"
            else:
                label = "Detecting"

            cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
            
    cv2.imshow('Webcam Feed', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()