from ultralytics import YOLO
import cv2
import os
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from fastapi import FastAPI
from fastapi.responses import JSONResponse
import uvicorn

load_dotenv()

# Load YOLOv8 model
model = YOLO("yolov8-weights/yolov8n.pt")

# Load BLIP captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Folder path
folder_path = "/Users/vedantprashantbhosale/Desktop/Akai visison craft/images"

app = FastAPI()

@app.get("/list_images")
def list_images():
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return JSONResponse(content={"error": "Folder path does not exist or is not a directory"}, status_code=400)

    supported_exts = (".jpg", ".jpeg", ".png")
    images = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.lower().endswith(supported_exts)]
    return {"images": images}

@app.post("/process_folder")
def process_folder():
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return JSONResponse(content={"error": "Folder path does not exist or is not a directory"}, status_code=400)

    all_outputs = []
    supported_exts = (".jpg", ".jpeg", ".png")

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(supported_exts):
            continue
        image_path = os.path.join(folder_path, filename)

        image_cv = cv2.imread(image_path)
        if image_cv is None:
            continue

        pil_image = Image.open(image_path).convert("RGB")

        # Run YOLOv8 detection
        results = model(image_cv)
        output_data = {
            "filePath": image_path,
            "annotations": [],
            "imageCaption": "",
            "status": "pending"
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

        all_outputs.append(output_data)

    return {"message": "Processing complete", "results": all_outputs}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
