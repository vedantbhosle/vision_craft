# Image Labeling Platform

An interactive web-based tool for **image annotation and caption editing**.
This platform combines **YOLOv8** for object detection, **BLIP** for automatic image captioning, and a **Streamlit UI** for editing bounding boxes and captions.

---

## 🚀 Features

* 🔍 **Automatic Object Detection** with YOLOv8
*  **AI-generated Captions** using BLIP
*  **Interactive Annotation Tool** with bounding box drawing
*  **Custom Label Assignment** for new bounding boxes
*  Save **updated captions** and **annotations** back to JSON
*  **Process entire folders** of images via FastAPI backend
*  Streamlit **frontend for editing & navigation**

---

## 🛠️ Tech Stack

* **Backend**: FastAPI, Uvicorn
* **Frontend**: Streamlit + Streamlit Drawable Canvas
* **AI Models**:

  * YOLOv8 (Ultralytics) for object detection
  * BLIP (Salesforce) for image captioning
* **Image Processing**: OpenCV, PIL
* **Data Storage**: JSON files for updated annotations and captions

---

## 📦 Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/yourusername/image-labeling-platform.git
   cd image-labeling-platform
   ```

2. **Create a virtual environment**

   ```bash
   python -m venv venv
   source venv/bin/activate   # Linux / macOS
   venv\Scripts\activate      # Windows
   ```
3. **Setup `.env` File**

    Create a `.env` file in the project root with the following:

    ```env
    HF_API_TOKEN=your_huggingface_api_token_here
    ```

    This is required for BLIP to fetch captions from Hugging Face.
4. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

5. **Download YOLOv8 weights**
   Place your YOLOv8 weights inside `yolov8-weights/` (e.g. `yolov8n.pt`).

6. **Set folder paths**
   Update `folder_path`, `annotated_folder`, and `updated_json_path` in the FastAPI backend (`main.py`).

---

## ▶️ Running the App

### 1. Start the FastAPI backend

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

### 2. Start the Streamlit frontend

```bash
streamlit run app.py
```

### 3. Workflow

1. Click **📂 Load & Process Images** in the Streamlit sidebar.
2. Browse images, navigate with **⬅️ Prev / Next ➡️**.
3. **Edit captions** or **draw bounding boxes**.

   * Only new bounding boxes will be stored.
   * Enter custom labels for each new box.
4. Save your updates → backend stores them in:

   * `updated_annotations.json` → bounding boxes
   * `updated_results.json` → captions

---

## 📁 Project Structure

```
.
├── client2.py                 # Streamlit frontend
├── app2.py                    # FastAPI backend
├── yolov8-weights/            # YOLOv8 model weights
├── images/                    # Input images
├── annotations/               # Auto-saved annotated images
├── updated_results.json       # Stores updated captions
├── updated_annotations.json   # Stores updated bounding boxes
├── requirements.txt           # Dependencies
└── README.md                  # Project documentation
```

---

## 📖 Example Usage

* **Draw a new bounding box** → A popup asks you for a label.
* **Save** → The label + coordinates are written into `updated_annotations.json`.
* **Edit caption** → Modify caption text and save → stored in `updated_results.json`.

---

## 🗄️ Example Screenshot

![alt text](image.png)
![alt text](image-1.png)
![alt text](image-2.png)
![alt text](image-3.png)
---

## 🤝 Contributing

Pull requests and issues are welcome.
For major changes, please open an issue first to discuss what you’d like to change.

---
