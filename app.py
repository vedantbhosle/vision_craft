from ultralytics import YOLO
import cv2
import numpy as np
import json
import os
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import JSONResponse
import uvicorn
from io import BytesIO

load_dotenv()

# Load YOLOv8 model
model = YOLO("yolov8-weights/yolov8n.pt")

# Load BLIP captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Create FastAPI app
app = FastAPI()

@app.post("/process_image")
async def process_image(image: UploadFile = File(None), image_path: str = Form(None)):
    if image is not None:
        file_bytes = await image.read()
        np_arr = np.frombuffer(file_bytes, np.uint8)
        image_cv = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        pil_image = Image.open(BytesIO(file_bytes)).convert("RGB")
        file_path_info = image.filename
    elif image_path is not None:
        if not os.path.exists(image_path):
            return JSONResponse(content={"error": "Image path does not exist"}, status_code=400)
        image_cv = cv2.imread(image_path)
        pil_image = Image.open(image_path).convert("RGB")
        file_path_info = image_path
    else:
        return JSONResponse(content={"error": "No image or image_path provided"}, status_code=400)

    if image_cv is None:
        return JSONResponse(content={"error": "Could not load image"}, status_code=400)

    # Run YOLOv8 detection
    results = model(image_cv)
    output_data = {
        "filePath": file_path_info,
        "annotations": [],
        "imageCaption": ""
    }

    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = model.names[int(box.cls[0])]
            output_data["annotations"].append({
                "classLabel": label,
                "bbox": [x1, y1, x2, y2]
            })

    # Run BLIP captioning
    inputs = processor(pil_image, return_tensors="pt")
    caption_ids = model_blip.generate(**inputs)
    caption = processor.decode(caption_ids[0], skip_special_tokens=True)
    output_data["imageCaption"] = caption

    return JSONResponse(content=output_data)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
