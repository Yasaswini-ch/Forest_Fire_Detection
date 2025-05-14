import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- CONFIG (VERIFIED FROM YOUR COLAB NOTEBOOK) ---
# Make sure these are correctly set based on your Colab training setup
# Example: If Colab's train_generator.class_indices was {'fire': 0, 'no_fire': 1}
# FIRE_CLASS_INDEX_FROM_COLAB = 0
# NO_FIRE_CLASS_INDEX_FROM_COLAB = 1
# Example: If Colab's train_generator.class_indices was {'no_fire': 0, 'fire': 1}
FIRE_CLASS_INDEX_FROM_COLAB = 1  # <<<--- KEEP THE CORRECT VALUE YOU DETERMINED
NO_FIRE_CLASS_INDEX_FROM_COLAB = 0 # <<<--- KEEP THE CORRECT VALUE YOU DETERMINED

TRAINING_IMG_WIDTH = 150  # <<<--- KEEP THE CORRECT VALUE
TRAINING_IMG_HEIGHT = 150 # <<<--- KEEP THE CORRECT VALUE
# --- END CONFIG ---


# Title of the web app
st.title("🔥 Forest Fire Detection App")

st.markdown("""
This Streamlit app uses a Convolutional Neural Network (CNN) to detect whether an uploaded image contains a **forest fire** or not.
""")
st.markdown("---")
st.subheader("Upload an Image")


# Load the trained model
@st.cache_resource
def load_my_model():
    try:
        loaded_model = tf.keras.models.load_model("FFD.keras")
        # st.success("Model loaded successfully!") # Optional: Can keep or remove
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model FFD.keras: {e}")
        st.error("Please ensure 'FFD.keras' is in the root directory and is a valid Keras model.")
        return None

model = load_my_model()

if model is None:
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Upload an image of a forest scene", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # st.markdown("---") # Optional: Can keep for visual separation or remove
    # st.subheader("Preprocessing Details (for debugging):") # REMOVE/COMMENT OUT

    # --- Preprocess image ---
    img_resized = image.resize((TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT))
    # st.write(f"1. Image resized to: {img_resized.size}") # REMOVE/COMMENT OUT

    img_array_pil = np.array(img_resized)
    # st.write(f"2a. PIL Image to NumPy array shape: {img_array_pil.shape}, dtype: {img_array_pil.dtype}") # REMOVE/COMMENT OUT
    # st.write(f"   Min/Max before normalization: {np.min(img_array_pil)}, {np.max(img_array_pil)}") # REMOVE/COMMENT OUT

    img_normalized = img_array_pil / 255.0
    # st.write(f"3. Normalized array shape: {img_normalized.shape}, dtype: {img_normalized.dtype}") # REMOVE/COMMENT OUT
    # st.write(f"   Min/Max after normalization: {np.min(img_normalized)}, {np.max(img_normalized)}") # REMOVE/COMMENT OUT

    img_batch = np.expand_dims(img_normalized, axis=0)
    # st.write(f"4. Batch array shape for model: {img_batch.shape}") # REMOVE/COMMENT OUT

    # --- Predict ---
    # st.markdown("---") # Optional: Can keep for visual separation or remove
    # st.subheader("Prediction Details (for debugging):") # REMOVE/COMMENT OUT
    
    with st.spinner("Classifying..."): # Keep the spinner for user experience
        raw_prediction = model.predict(img_batch)

    # st.write(f"Raw model output (prediction): {raw_prediction}") # REMOVE/COMMENT OUT
    
    predicted_value = raw_prediction[0][0]
    # st.write(f"Extracted scalar probability: {predicted_value:.4f}") # REMOVE/COMMENT OUT

    # --- Interpret Prediction ---
    final_result_text = ""
    final_verdict_style = st.error # Default to error style for 'Fire'

    if FIRE_CLASS_INDEX_FROM_COLAB == 1: # Model outputs P(fire)
        # st.write(f"Interpreting: Model outputs P(Fire). Threshold is 0.5. P(Fire) = {predicted_value:.4f}") # REMOVE/COMMENT OUT
        if predicted_value > 0.5:
            final_result_text = "🔥 Fire Detected!"
            final_verdict_style = st.error
        else:
            final_result_text = "✅ No Fire Detected."
            final_verdict_style = st.success
    elif FIRE_CLASS_INDEX_FROM_COLAB == 0: # Model outputs P(no_fire)
        # st.write(f"Interpreting: Model outputs P(No Fire). Threshold is 0.5. P(No Fire) = {predicted_value:.4f}") # REMOVE/COMMENT OUT
        if predicted_value > 0.5: # High probability of 'No Fire'
            final_result_text = "✅ No Fire Detected."
            final_verdict_style = st.success
        else: # Low probability of 'No Fire' means high probability of 'Fire'
            final_result_text = "🔥 Fire Detected!"
            final_verdict_style = st.error
    else:
        # This case should ideally not be hit if config is correct
        st.warning("Configuration error: FIRE_CLASS_INDEX_FROM_COLAB is not set correctly.")
        final_result_text = "Error in app configuration"
        final_verdict_style = st.warning


    # st.markdown("---") # Optional: Can keep for visual separation or remove
    st.subheader("Final Verdict:") # Keep this subheader
    final_verdict_style(final_result_text) # Use the determined style (error or success)

else:
    st.info("Please upload an image file.")

st.markdown("---")
st.markdown("Developed by Yasaswini Chebolu")
st.markdown("Check out the [GitHub Repository](https://github.com/Yasaswini-ch/Forest_Fire_Detection)") # Replace with your actual repo link
