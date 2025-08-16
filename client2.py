import streamlit as st
import requests
from PIL import Image

st.set_page_config(layout="wide")
st.title("AkaiVisionCraft - Annotation Viewer")

# --- Configuration ---
API_URL_PROCESS = "http://localhost:8000/process_folder"
API_URL_UPDATE = "http://localhost:8000/update_caption"

# --- Session State Initialization ---
if "results" not in st.session_state:
    st.session_state.results = []
if "current_index" not in st.session_state:
    st.session_state.current_index = 0
if "edit_mode" not in st.session_state:
    st.session_state.edit_mode = {}  # Store per-image edit mode state

# --- Main Application Logic ---
if st.button("Load & Process Images", use_container_width=True):
    with st.spinner("Processing images via API... Please wait."):
        try:
            res = requests.post(API_URL_PROCESS)
            if res.status_code == 200:
                st.session_state.results = res.json().get("results", [])
                st.session_state.current_index = 0
                if not st.session_state.results:
                    st.warning("No results returned from API.")
                else:
                    st.success(f"Successfully loaded {len(st.session_state.results)} images!")
            else:
                st.error(f"API Error {res.status_code}: {res.text}")
        except requests.exceptions.RequestException as e:
            st.error(f"Failed to connect to the backend at {API_URL_PROCESS}.")
            st.error(f"Error: {e}")
            st.info("Please make sure your FastAPI backend script is running.")

# Display content if results are available
if st.session_state.results:
    total_images = len(st.session_state.results)
    entry = st.session_state.results[st.session_state.current_index]
    file_path = entry["filePath"]

    # Ensure each image has its own edit mode state
    if file_path not in st.session_state.edit_mode:
        st.session_state.edit_mode[file_path] = False

    st.markdown("---")
    main_col1, main_col2 = st.columns([2, 1])

    # Display images
    with main_col1:
        st.subheader(f"Image {st.session_state.current_index + 1} of {total_images}")
        img_col1, img_col2 = st.columns(2)
        try:
            with img_col1:
                st.markdown("##### Original Image")
                original_image = Image.open(entry["filePath"]).convert("RGB")
                st.image(original_image, use_container_width=True)

            with img_col2:
                st.markdown("##### Annotated Image")
                annotated_image = Image.open(entry["annotatedFilePath"]).convert("RGB")
                st.image(annotated_image, use_container_width=True)
        except FileNotFoundError as e:
            st.error(f"Image file not found: {e.filename}")
        except Exception as e:
            st.error(f"An error occurred while loading an image: {e}")

    # Caption + Edit control
    with main_col2:
        st.subheader("AI-Generated Caption")

        # Yes/No buttons to enable/disable editing per image
        col_yes, col_no = st.columns(2)
        with col_yes:
            if st.button(" Yes (Edit Caption)"):
                st.session_state.edit_mode[file_path] = True
        with col_no:
            if st.button(" No (Keep Caption)"):
                st.session_state.edit_mode[file_path] = False

        current_caption = entry.get('imageCaption', 'Not available')

        if st.session_state.edit_mode[file_path]:
            # Editable caption
            edited_caption = st.text_area(
                label="Edit the caption below:",
                value=current_caption,
                key=f"caption_{st.session_state.current_index}"
            )
            if edited_caption != current_caption:
                # Update frontend state
                st.session_state.results[st.session_state.current_index]['imageCaption'] = edited_caption

                # Call API to update backend JSON
                try:
                    update_res = requests.post(API_URL_UPDATE, json={
                        "filePath": file_path,
                        "newCaption": edited_caption
                    })
                    if update_res.status_code == 200:
                        st.toast("✅ Caption updated in backend!")
                    else:
                        st.error(f"Backend update failed: {update_res.text}")
                except requests.exceptions.RequestException as e:
                    st.error(f"Error updating caption in backend: {e}")
        else:
            # Read-only caption
            st.info(f"_{current_caption}_")
            
        # Navigation
        st.markdown("---")
        st.subheader("Navigation")
        prev_col, next_col = st.columns(2)
        with prev_col:
            if st.button("⬅️ Previous", use_container_width=True):
                st.session_state.current_index = (st.session_state.current_index - 1 + total_images) % total_images
                st.rerun()
        with next_col:
            if st.button("Next ➡️", use_container_width=True):
                st.session_state.current_index = (st.session_state.current_index + 1) % total_images
                st.rerun()

else:
    st.info("Click 'Load & Process Images' to fetch and display data from your API.")