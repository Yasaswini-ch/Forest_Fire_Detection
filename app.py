import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- DEBUGGING FLAGS & CONFIG (YOU NEED TO VERIFY THESE FROM YOUR COLAB NOTEBOOK) ---
# In Colab, check: train_generator.class_indices
# Example: If {'fire': 0, 'no_fire': 1} then:
# FIRE_CLASS_INDEX_FROM_COLAB = 0
# NO_FIRE_CLASS_INDEX_FROM_COLAB = 1
# Example: If {'no_fire': 0, 'fire': 1} then:
FIRE_CLASS_INDEX_FROM_COLAB = 1 # <<<--- IMPORTANT: SET THIS BASED ON YOUR COLAB
NO_FIRE_CLASS_INDEX_FROM_COLAB = 0 # <<<--- IMPORTANT: SET THIS BASED ON YOUR COLAB

# What were the exact image dimensions used for training in Colab?
TRAINING_IMG_WIDTH = 150 # <<<--- VERIFY: Your current code uses 150
TRAINING_IMG_HEIGHT = 150 # <<<--- VERIFY: Your current code uses 150
# --- END DEBUGGING FLAGS & CONFIG ---


# Title of the web app
st.title("🔥 Forest Fire Detection App")

st.markdown("""
This Streamlit app uses a Convolutional Neural Network (CNN) to detect whether an uploaded image contains a **forest fire** or not.
""")
st.markdown("---")
st.subheader("Upload an Image")


# Load the trained model
@st.cache_resource
def load_my_model(): # Renamed to avoid potential conflicts if 'load_model' is a common name
    try:
        loaded_model = tf.keras.models.load_model("FFD.keras")
        st.success("Model loaded successfully!")
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model FFD.keras: {e}")
        st.error("Please ensure 'FFD.keras' is in the root directory and is a valid Keras model.")
        return None

model = load_my_model()

if model is None:
    st.stop() # Stop the app if the model couldn't be loaded

# Upload image
uploaded_file = st.file_uploader("Upload an image of a forest scene", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    st.markdown("---")
    st.subheader("Preprocessing Details (for debugging):")

    # --- Preprocess image ---
    # 1. Resize
    # IMPORTANT: Ensure this matches the input size your model was trained on in Colab.
    img_resized = image.resize((TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT))
    st.write(f"1. Image resized to: {img_resized.size}")

    # 2. Convert to NumPy array
    # IMPORTANT: In your original code, you did np.array(img) / 255.0.
    # It's generally better to use tf.keras.preprocessing.image.img_to_array
    # as it handles channel ordering correctly for Keras models (channels last by default).
    # However, for simple RGB images from Pillow, np.array() is usually fine.
    # Let's stick to your method for now and add debugging.
    
    img_array_pil = np.array(img_resized)
    st.write(f"2a. PIL Image to NumPy array shape: {img_array_pil.shape}, dtype: {img_array_pil.dtype}")
    st.write(f"   Min/Max before normalization: {np.min(img_array_pil)}, {np.max(img_array_pil)}")

    # 3. Normalize
    # IMPORTANT: Ensure this matches the normalization used during training.
    # If you used a specific preprocess_input function (e.g., for MobileNet) in Colab, replicate that here.
    # Assuming simple /255.0 scaling as in your code.
    img_normalized = img_array_pil / 255.0
    st.write(f"3. Normalized array shape: {img_normalized.shape}, dtype: {img_normalized.dtype}")
    st.write(f"   Min/Max after normalization: {np.min(img_normalized)}, {np.max(img_normalized)}")
    # st.write(f"   Sample normalized pixel (top-left): {img_normalized[0,0,:]}") # If you want to see pixel values

    # 4. Expand dimensions for batch
    img_batch = np.expand_dims(img_normalized, axis=0)
    st.write(f"4. Batch array shape for model: {img_batch.shape}")

    # --- Predict ---
    st.markdown("---")
    st.subheader("Prediction Details (for debugging):")
    
    with st.spinner("Classifying..."):
        raw_prediction = model.predict(img_batch)

    st.write(f"Raw model output (prediction): {raw_prediction}")
    # raw_prediction is likely [[some_float_value]], e.g., [[0.123]] or [[0.897]]
    
    predicted_value = raw_prediction[0][0] # Extract the scalar probability
    st.write(f"Extracted scalar probability: {predicted_value:.4f}")

    # --- Interpret Prediction ---
    # This logic depends ENTIRELY on your Colab's train_generator.class_indices
    # If FIRE_CLASS_INDEX_FROM_COLAB is 0, it means the model outputs the probability of the OTHER class (NO_FIRE_CLASS_INDEX_FROM_COLAB=1)
    # If FIRE_CLASS_INDEX_FROM_COLAB is 1, it means the model outputs the probability of THIS class (FIRE_CLASS_INDEX_FROM_COLAB=1)

    # Let's assume the model outputs the probability of the class that was assigned index 1 during training.
    # So, if 'fire' was index 1, `predicted_value` is P(fire).
    # If 'no_fire' was index 1, `predicted_value` is P(no_fire).

    final_result_text = ""
    if FIRE_CLASS_INDEX_FROM_COLAB == 1: # Model outputs P(fire)
        st.write(f"Interpreting: Model outputs P(Fire). Threshold is 0.5. P(Fire) = {predicted_value:.4f}")
        if predicted_value > 0.5:
            final_result_text = "🔥 Fire Detected!"
            st.error(final_result_text)
        else:
            final_result_text = "✅ No Fire Detected."
            st.success(final_result_text)
    elif FIRE_CLASS_INDEX_FROM_COLAB == 0: # Model outputs P(no_fire) because 'fire' was index 0
        st.write(f"Interpreting: Model outputs P(No Fire). Threshold is 0.5. P(No Fire) = {predicted_value:.4f}")
        if predicted_value > 0.5: # High probability of 'No Fire'
            final_result_text = "✅ No Fire Detected."
            st.success(final_result_text)
        else: # Low probability of 'No Fire' means high probability of 'Fire'
            final_result_text = "🔥 Fire Detected!"
            st.error(final_result_text)
    else:
        st.warning("FIRE_CLASS_INDEX_FROM_COLAB is not set to 0 or 1. Cannot interpret results.")
        final_result_text = "Error in configuration"

    # Your original interpretation (simpler, but relies on knowing what >0.5 means):
    # class_names = ["No Fire", "Fire"] # This implies 'No Fire' is 0, 'Fire' is 1 for the output if > 0.5 means 'Fire'
    # result = class_names[int(prediction[0][0] > 0.5)]
    # st.subheader(f"🔍 Prediction: **{result}**")
    st.markdown("---")
    st.subheader(f"Final Verdict: {final_result_text}")

else:
    st.info("Please upload an image file.")

st.markdown("---")
st.markdown("Developed by Yasaswini Chebolu")
st.markdown("Check out the [GitHub Repository](https://github.com/Yasaswini-ch/Forest_Fire_Detection)") # Replace with your actual repo link
