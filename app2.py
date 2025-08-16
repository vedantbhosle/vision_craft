from ultralytics import YOLO
import cv2
import os
import json
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import uvicorn

load_dotenv()

# YOLOv8 model
model = YOLO("yolov8-weights/yolov8n.pt")

# BLIP captioning model
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

# Folders (upload your folder paths)
folder_path = "/Users/vedantprashantbhosale/Desktop/Akai visison craft/images"
annotated_folder = "/Users/vedantprashantbhosale/Desktop/Akai visison craft/annotations"
updated_json_path = "/Users/vedantprashantbhosale/Desktop/Akai visison craft/updated_results.json"
os.makedirs(annotated_folder, exist_ok=True)

last_results = []

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
    global last_results
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

        # Run YOLOv8 detection
        pil_image = Image.open(image_path).convert("RGB")
        results = model(image_cv)
        output_data = {
            "filePath": image_path,
            "annotatedFilePath": "",
            "annotations": [],
            "imageCaption": "",
            "status": "pending"
        }

        for result in results:
            boxes = result.boxes
            for box in boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                label = model.names[int(box.cls[0])]
                conf = float(box.conf[0])
                cv2.rectangle(image_cv, (x1, y1), (x2, y2), (0, 255, 0), 4)
                cv2.putText(image_cv, f"{label} {conf:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                output_data["annotations"].append({"classLabel": label, "bbox": [x1, y1, x2, y2]})

        annotated_path = os.path.join(annotated_folder, filename)
        cv2.imwrite(annotated_path, image_cv)
        output_data["annotatedFilePath"] = annotated_path

        inputs = processor(pil_image, return_tensors="pt")
        caption_ids = model_blip.generate(**inputs)
        caption = processor.decode(caption_ids[0], skip_special_tokens=True)
        output_data["imageCaption"] = caption

        all_outputs.append(output_data)

    # Merge with any previous results to retain updated captions
    if os.path.exists(updated_json_path):
        try:
            with open(updated_json_path, "r") as f:
                prev_data = json.load(f)
            for prev_item in prev_data:
                for new_item in all_outputs:
                    if prev_item["filePath"] == new_item["filePath"]:
                        new_item["imageCaption"] = prev_item.get("imageCaption", new_item["imageCaption"])
        except Exception:
            pass

    last_results = all_outputs

    return {"message": "Processing complete", "results": all_outputs}

@app.post("/update_caption")
async def update_caption(request: Request):
    global last_results
    data = await request.json()
    file_path = data.get("filePath")
    new_caption = data.get("newCaption")
    updated = False

    for item in last_results:
        if item["filePath"] == file_path:
            item["imageCaption"] = new_caption
            item["status"] = "updated"  
            updated = True
            break

    if updated:
        with open(updated_json_path, "w") as f:
            json.dump(last_results, f, indent=4)
        return {"message": "Caption updated successfully", "results": last_results}
    else:
        return JSONResponse(content={"error": "File path not found in last results"}, status_code=404)

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)