import streamlit as st
from PIL import Image
import io
import numpy as np
import cv2
from model import ALPR

# Import your model here
my_model = ALPR()

st.title('License Plate Recognition')
st.write('Upload an image and the model will detect and read the license plate on UFPR-ALPR Dataset preferably testing set.')
st.write('Made by: [Andrew Willy](https://github.com/Sebercheres/licence-plate-recognition)')

# Image uploader
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
if uploaded_file is not None:
    # To read file as bytes:
    bytes_data = uploaded_file.getvalue()
    image = Image.open(io.BytesIO(bytes_data))

    # Convert PIL Image to numpy array
    image_np = np.array(image.convert('RGB'))

    # Display the uploaded image
    st.image(image, caption='Uploaded Image.', use_column_width=True)

    # Predict or process the input
    if st.button('Run Model'):
        # Use the forward method of ALPR class
        bbox, cropped_img, ocr_result = my_model.forward(image_np)

        if bbox and cropped_img is not None:
            # Draw bounding box on the original image
            x1, y1, x2, y2 = bbox
            cv2.rectangle(image_np, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)

            # Convert numpy array back to PIL Image for display
            boxed_image = Image.fromarray(image_np)
            cropped_image = Image.fromarray(cropped_img)

            # Display the images with bounding box and the cropped image
            st.image(boxed_image, caption="Image with Detected Region", use_column_width=True)
            st.image(cropped_image, caption="Cropped Image", use_column_width=True)

            # Display the OCR result
            text, confidence = ocr_result[0][0]
            st.write("OCR Result: ", text)
            st.write("Confidence: ", confidence)
        else:
            st.write("No license plate detected.")
