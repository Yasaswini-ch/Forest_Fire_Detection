import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# Title of the web app
st.title("🔥 Forest Fire Detection App")

st.markdown("""
This Streamlit app uses a Convolutional Neural Network (CNN) to detect whether an uploaded image contains a **forest fire** or not.
""")

# Load the trained model
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model("FFD.keras")
    return model

model = load_model()

# Upload image
uploaded_file = st.file_uploader("Upload an image of a forest scene", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Classifying...")

    # Preprocess image
    img = image.resize((150, 150))
    img_array = np.array(img) / 255.0  # normalize
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    prediction = model.predict(img_array)
    class_names = ["No Fire", "Fire"]
    result = class_names[int(prediction[0][0] > 0.5)]

    st.subheader(f"🔍 Prediction: **{result}**")
