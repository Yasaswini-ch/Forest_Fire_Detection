import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

# --- CONFIG (VERIFY THESE FROM YOUR COLAB NOTEBOOK) ---
# These TRAINING_IMG values MUST match your Colab training.
TRAINING_IMG_WIDTH = 150  # Example value, update if different
TRAINING_IMG_HEIGHT = 150 # Example value, update if different
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
        return loaded_model
    except Exception as e:
        st.error(f"Error loading model FFD.keras: {e}")
        return None

model = load_my_model()

if model is None:
    st.stop()

# Upload image
uploaded_file = st.file_uploader("Upload an image of a forest scene", type=["jpg", "jpeg", "png"], label_visibility="collapsed")

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # --- Preprocess image (Keep this consistent with your training) ---
    img_resized = image.resize((TRAINING_IMG_WIDTH, TRAINING_IMG_HEIGHT))
    img_array_pil = np.array(img_resized)
    img_normalized = img_array_pil / 255.0 # Assuming simple /255.0 normalization
    img_batch = np.expand_dims(img_normalized, axis=0)

    # --- Predict ---
    with st.spinner("Classifying..."):
        raw_prediction = model.predict(img_batch)
    
    predicted_value = raw_prediction[0][0]
    #st.write(f"DEBUG: Raw scalar prediction from model: {predicted_value:.4f}") # DEBUG LINE

    # --- Interpret Prediction ---
    # CHOOSE *ONE* OF THE FOLLOWING INTERPRETATION BLOCKS TO TEST AT A TIME.
    # COMMENT OUT THE OTHER ONE.

    # ============================================================================
    # == TEST INTERPRETATION A: Assume `> 0.5` means "FIRE" ==
    # ============================================================================
    final_result_text_A = ""
    final_verdict_style_A = st.error 

    st.markdown("---") # Separator for clarity
    #st.write("DEBUG: Testing Interpretation A: `< 0.5` means FIRE") 
    if predicted_value < 0.5:
        final_result_text_A = "🔥 Fire Detected!"
        final_verdict_style_A = st.error
    else:
        final_result_text_A = "✅ No Fire Detected."
        final_verdict_style_A = st.success
    
    #st.subheader("Final Verdict (Using Interpretation A):")
    final_verdict_style_A(final_result_text_A)
    # ============================================================================
    # == END OF TEST INTERPRETATION A ==
    # =================================================_==_===========================


    # ============================================================================
    # == TEST INTERPRETATION B: Assume `< 0.5` means "FIRE" (i.e., >0.5 is NO FIRE) ==
    # ============================================================================
    # final_result_text_B = ""
    # final_verdict_style_B = st.error

    # st.markdown("---") # Separator for clarity
    # st.write("DEBUG: Testing Interpretation B: `< 0.5` means FIRE")
    # if predicted_value < 0.5: # Note the change here: < 0.5 for FIRE
    #     final_result_text_B = "🔥 Fire Detected! (Interp B)"
    #     final_verdict_style_B = st.error
    # else:
    #     final_result_text_B = "✅ No Fire Detected. (Interp B)"
    #     final_verdict_style_B = st.success

    # st.subheader("Final Verdict (Using Interpretation B):")
    # final_verdict_style_B(final_result_text_B)
    # ============================================================================
    # == END OF TEST INTERPRETATION B ==
    # ============================================================================

else:
    st.info("Please upload an image file.")

st.markdown("---")
st.markdown("Developed by Yasaswini Chebolu")
st.markdown("Check out the [GitHub Repository](https://github.com/Yasaswini-ch/Forest_Fire_Detection)")
