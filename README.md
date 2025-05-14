# Forest Fire Detection Using CNN(Week-1,2,3)

## Forest Fire Detection (Week 1 & 2)

## Overview

This project aims to build a deep learning model for detecting forest fires from images. Forest fires pose a significant risk to both human lives and the environment. Detecting these fires early is crucial for mitigating the damage. The goal of this project is to create an image classification model using Convolutional Neural Networks (CNN) to identify fire and no-fire images from a dataset of forest fire images.

### Week 1 Work:
- Setup and exploration of the **Wildfire Dataset**.
- Organizing the dataset into training, validation, and testing directories.
- Visualizing a few sample images from both the "fire" and "no fire" classes.
- Basic understanding of the dataset structure.

## Dataset

The dataset used for this project is the **Wildfire Dataset**. It contains images of forested areas in two classes:

- **Fire**: Images of forest fires.
- **No Fire**: Images of forest areas without any fire.

The dataset is organized into three main directories:

- **Train**: Used for training the model.
- **Validation**: Used to evaluate the model during training.
- **Test**: Used to evaluate the final performance of the model.

## Project Structure

1. **Dataset Exploration**:
   - Explore the dataset to check the images and their distribution across classes.

2. **Model Construction**:
   - Use a Convolutional Neural Network (CNN) for classifying the images into fire or no-fire categories.

3. **Data Preprocessing**:
   - Images are resized and rescaled to prepare them for training.
   - Data generators are used to load the images in batches, which helps in better memory management and faster training.

4. **Training the Model**:
   - The model will be trained using the training dataset, and its performance will be evaluated using the validation and test datasets in later stages.

## Steps Involved

1. **Install Required Libraries**: 
   Install necessary libraries and set up the environment for the project.

2. **Dataset Setup**:
   Download the dataset from Kaggle and organize it into the required directories (train, validation, and test).

3. **Visualize the Data**:
   Visualize a few sample images from both classes to understand the dataset better.

## Technologies Used

- **TensorFlow/Keras**: A deep learning framework used to build and train the CNN model.
- **Kaggle**: For accessing and downloading the dataset.
- **Google Colab**: Used for running the code in the cloud with GPU support.
- **Matplotlib**: Used for visualizing the sample images from the dataset.
# Forest Fire Detection (Week 3)

## Overview

In Week 3, the focus shifted from model training to integration and deployment. The trained CNN model was connected to a Streamlit web interface to allow real-time fire detection from user-uploaded images. The deployment pipeline was also finalized for hosting the project via GitHub and Streamlit Cloud.

## Week 3 Work:

###  Model Saving and Finalization
- Completed training and validation of the CNN model.
- Saved the trained model in `.keras` format as `FFD.keras` for easy loading and reuse.

###  Streamlit App Development (`app.py`)
- Created a responsive web interface using Streamlit.
- Features of the app:
  - Upload images directly through the browser.
  - Preview of uploaded image.
  - Predicts whether the uploaded image shows **Fire** or **No Fire**.
  - Displays prediction confidence score.
- Integrated `FFD.keras` model into the app using `tensorflow.keras.models.load_model()`.
  
**Features:**
- Upload and preview images
- Predict and display fire/no-fire result
- User-friendly interface for real-time interaction

###  Repository Structure Finalized
The following files were organized and pushed to GitHub:
```
forest-fire-detection/
├── FFD.keras            # Saved model
├── app.py               # Streamlit app
├── requirements.txt     # Dependencies
├── README.md            # Documentation  
└── wildfire-dataset/    # Organized dataset (train/val/test)
```
## Deployment Guide

The application can be hosted via GitHub and Streamlit Cloud:

1. Push the following files to your GitHub repository:
   - `app.py`
   - `FFD.keras`
   - `requirements.txt`
   - `README.md`
2. Go to [streamlit.io/cloud](https://streamlit.io/cloud)
3. Connect your GitHub repository
4. Set `app.py` as the main file
5. Launch the application

---

## requirements.txt

```txt
streamlit
tensorflow
pillow
numpy
```

---

## Technologies Used

| Tool             | Purpose                             |
|------------------|--------------------------------------|
| Python           | Core programming language            |
| TensorFlow/Keras | Deep learning model development      |
| NumPy            | Numerical computations               |
| Matplotlib       | Visualization of data and results    |
| Streamlit        | Application interface development    |
| Google Colab     | Cloud-based model training           |
| GitHub           | Version control and deployment       |

---

## Future Enhancements

- Add webcam-based real-time fire detection capabilities
- Improve model performance with additional data and advanced regularization
- Incorporate Grad-CAM visualizations for interpretability
- Implement real-time alert systems (e.g., SMS, email notifications)

---
