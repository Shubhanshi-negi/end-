from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import cv2
import numpy as np
import base64

app = FastAPI()
model = YOLO("yolov8n.pt")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image_bytes = await file.read()
    np_img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

    results = model(img)
    annotated = results[0].plot()

    _, buffer = cv2.imencode(".jpg", annotated)

    encoded = base64.b64encode(buffer).decode("utf-8")
    return {"image": encoded}
