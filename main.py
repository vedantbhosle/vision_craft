from ultralytics import YOLO
import cv2
import json
import os
from dotenv import load_dotenv
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image

load_dotenv()

model = YOLO("yolov8-weights/yolov8n.pt")


# ---- Load BLIP captioning model ----
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model_blip = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")

image_path = "images/sample4.jpg"
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"Image not found: {image_path}")


results = model(image)


output_data = {
    "filePath": image_path,
    "annotations": [],
    "imageCaption": ""  
}


for result in results:
    boxes = result.boxes
    for box in boxes:
        # Bounding box
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        # Class label
        label = model.names[int(box.cls[0])]
        # Confidence
        conf = float(box.conf[0])

        # Draw on image (optional)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 4)
        cv2.putText(image, f"{label} {conf:.2f}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        # Append to JSON annotations
        output_data["annotations"].append({
            "classLabel": label,
            "bbox": [x1, y1, x2, y2]
        })

pil_image = Image.open(image_path).convert("RGB")
inputs = processor(pil_image, return_tensors="pt")
caption_ids = model_blip.generate(**inputs)
caption = processor.decode(caption_ids[0], skip_special_tokens=True)
output_data["imageCaption"] = caption


# Show image with detections
cv2.imshow("YOLOv8 Detection", image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# JSON output
output_file = os.path.splitext(os.path.basename(image_path))[0] + ".json"
with open(output_file, "w") as f:
    json.dump(output_data, f, indent=4)

print(f"Saved annotations to {output_file}")