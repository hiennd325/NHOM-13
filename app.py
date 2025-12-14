import streamlit as st
import cv2
import numpy as np
from PIL import Image

st.title("Nhận diện biển số xe nhóm 13")

st.write("Tải ảnh lên")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Read the image
    image = Image.open(uploaded_file)
    img_array = np.array(image)
    img_cv = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)

    # Process the image
    with st.spinner("Processing..."):
        img_with_boxes, plates = process_image(img_cv)

    # Display original with boxes
    st.subheader("Image with Detected Plates")
    img_with_boxes_rgb = cv2.cvtColor(img_with_boxes, cv2.COLOR_BGR2RGB)
    st.image(img_with_boxes_rgb, use_column_width=True)

    # Display plates
    if plates:
        st.subheader("Recognized License Plates")
        for i, (plate_img, text) in enumerate(plates):
            st.write(f"**Plate {i+1}:** {text}")
            plate_img_rgb = cv2.cvtColor(plate_img, cv2.COLOR_BGR2RGB)
            st.image(plate_img_rgb, width=300)
    else:
        st.write("No license plates detected.")