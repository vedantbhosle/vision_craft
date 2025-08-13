import streamlit as st
import requests

st.title("YOLOv8 + BLIP Image Processing Client")

backend_url = st.text_input("Backend API URL", "http://localhost:8000/process_image")

upload_option = st.radio("Choose input type", ("Upload Image", "Image Path"))

if upload_option == "Upload Image":
    uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)
        if st.button("Process Image"):
            files = {"image": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
            response = requests.post(backend_url, files=files)
            if response.status_code == 200:
                st.json(response.json())
            else:
                st.error(f"Error: {response.status_code} - {response.text}")

elif upload_option == "Image Path":
    image_path = st.text_input("Enter the image path accessible to backend")
    if st.button("Process Image") and image_path:
        data = {"image_path": image_path}
        response = requests.post(backend_url, data=data)
        if response.status_code == 200:
            st.json(response.json())
        else:
            st.error(f"Error: {response.status_code} - {response.text}")
