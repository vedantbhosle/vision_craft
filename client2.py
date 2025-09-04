import streamlit as st
import requests
from streamlit_drawable_canvas import st_canvas
from PIL import Image

st.set_page_config(layout="wide")
st.title("AkaiVisionCraft - Editable Bounding Box & Caption Editor")

# --- Configuration ---
API_URL_PROCESS = "http://localhost:8000/process_folder"
API_URL_UPDATE_CAPTION = "http://localhost:8000/update_caption"
API_URL_UPDATE_BBOX = "http://localhost:8000/update_bbox"

# --- Session State Initialization ---
if "results" not in st.session_state:
    st.session_state.results = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "new_boxes" not in st.session_state:
    st.session_state.new_boxes = {}

# --- Sidebar controls ---
st.sidebar.header("Controls")
if st.sidebar.button("📂 Load & Process Images", use_container_width=True):
    with st.spinner("Fetching images and annotations from API..."):
        try:
            res = requests.post(API_URL_PROCESS)
            if res.status_code == 200:
                st.session_state.results = res.json().get("results", [])
                st.session_state.current_index = 0
                st.session_state.new_boxes = {}
                if not st.session_state.results:
                    st.warning("No results returned from API.")
                else:
                    st.success(f"✅ Loaded {len(st.session_state.results)} images!")
            else:
                st.error(f"API Error {res.status_code}: {res.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to backend: {e}")

# --- Navigation ---
if st.session_state.results:
    total_images = len(st.session_state.results)
    prev_col, next_col = st.sidebar.columns(2)
    with prev_col:
        if st.button("⬅️ Prev", use_container_width=True):
            st.session_state.current_index = (st.session_state.current_index - 1 + total_images) % total_images
            st.rerun()
    with next_col:
        if st.button("Next ➡️", use_container_width=True):
            st.session_state.current_index = (st.session_state.current_index + 1) % total_images
            st.rerun()

# --- Display image and caption editor ---
if st.session_state.results:
    entry = st.session_state.results[st.session_state.current_index]
    file_path = entry["filePath"]

    if file_path not in st.session_state.new_boxes:
        st.session_state.new_boxes[file_path] = []

    st.subheader(f"🖼️ Image {st.session_state.current_index + 1} of {len(st.session_state.results)}")
    col1, col2 = st.columns([2, 1])

    with col1:
        try:
            img = Image.open(file_path).convert("RGB")

            # Resize image if too large (max width = 800px)
            max_width = 800
            if img.width > max_width:
                ratio = max_width / img.width
                new_size = (max_width, int(img.height * ratio))
                img = img.resize(new_size)

            # Do NOT load old annotations (only allow drawing new ones)
            canvas_result = st_canvas(
                fill_color="rgba(0, 0, 0, 0)",
                stroke_color="green",
                stroke_width=3,
                background_image=img,
                update_streamlit=True,
                width=img.width,
                height=img.height,
                drawing_mode="rect",
                key=f"canvas_{st.session_state.current_index}",
            )

            new_boxes = []
            if canvas_result.json_data is not None:
                for i, obj in enumerate(canvas_result.json_data["objects"]):
                    if obj["type"] == "rect":
                        x1, y1 = obj["left"], obj["top"]
                        x2, y2 = x1 + obj["width"], y1 + obj["height"]

                        # Ask for label for each NEW drawn box
                        label = st.text_input(f"Label for Box {i+1}", key=f"label_{file_path}_{i}")

                        new_boxes.append({
                            "classLabel": label if label else "object",
                            "bbox": [int(x1), int(y1), int(x2), int(y2)]
                        })

                st.session_state.new_boxes[file_path] = new_boxes

            if st.button("💾 Save New Boxes"):
                try:
                    update_res = requests.post(API_URL_UPDATE_BBOX, json={
                        "filePath": file_path,
                        "annotations": st.session_state.new_boxes[file_path]
                    })
                    if update_res.status_code == 200:
                        st.success("✅ New boxes saved!")
                        st.session_state.new_boxes[file_path] = []  # Clear after save
                    else:
                        st.error(f"Failed to update: {update_res.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error updating boxes: {e}")

        except Exception as e:
            st.error(f"Error loading image: {e}")

    with col2:
        st.subheader("📝 Caption Editor")
        current_caption = entry.get("imageCaption", "Not available")

        if st.toggle("Edit Caption", key=f"toggle_{file_path}"):
            edited_caption = st.text_area("Caption", value=current_caption, key=f"caption_{st.session_state.current_index}", label_visibility="collapsed")
            if st.button("💾 Save Caption"):
                try:
                    update_res = requests.post(API_URL_UPDATE_CAPTION, json={
                        "filePath": file_path,
                        "newCaption": edited_caption
                    })
                    if update_res.status_code == 200:
                        st.success("✅ Caption updated!")
                    else:
                        st.error(f"Failed to update caption: {update_res.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error updating caption: {e}")
        else:
            st.info(f"_{current_caption}_")
else:
    st.info("Click '📂 Load & Process Images' to begin.")