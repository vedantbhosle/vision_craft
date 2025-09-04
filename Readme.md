# 📸 Image Labeling Platform

An interactive web-based tool for **image annotation and caption editing**.  
This platform combines **YOLOv8** for object detection, **BLIP** for automatic image captioning, and a **Streamlit UI** for editing bounding boxes and captions.  

---

## 🚀 Features
- 🔍 **Automatic Object Detection** with YOLOv8  
- 📝 **AI-generated Captions** using BLIP  
- 🎨 **Interactive Annotation Tool** with bounding box drawing  
- ✏️ **Custom Label Assignment** for new bounding boxes  
- 💾 Save **updated captions** and **annotations** back to JSON  
- 📂 **Process entire folders** of images via FastAPI backend  
- 🌐 Streamlit **frontend for editing & navigation**  

---

## 🛠️ Tech Stack
- **Backend**: FastAPI, Uvicorn  
- **Frontend**: Streamlit + Streamlit Drawable Canvas  
- **AI Models**:  
  - YOLOv8 (Ultralytics) for object detection  
  - BLIP (Salesforce) for image captioning  
- **Image Processing**: OpenCV, PIL  
- **Data Storage**: JSON files for updated annotations and captions  

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/image-labeling-platform.git
   cd image-labeling-platform
