import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load trained model
model = load_model("facemask_VGG16.h5")

st.title("😷 Face Mask Detection App")
st.write("Upload an image to check whether a person is wearing a mask or not.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Preprocess image
    img = image.resize((200, 200))
    img = np.array(img)

    # ✅ FIX: Convert RGB (PIL) → BGR (cv2) to match training data
    img = img[:, :, ::-1]

    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)
    st.write("Prediction Probabilities:", prediction)

    class_names = ["No Mask", "Mask"]
    predicted_class = np.argmax(prediction)

    if class_names[predicted_class] == "Mask":
        st.success("✅ Mask Detected")
    else:
        st.error("❌ No Mask Detected")