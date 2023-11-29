import streamlit as st
import numpy as np
import tensorflow as tf
from keras.applications.vgg16 import preprocess_input
import joblib
import cv2
from skimage.metrics import structural_similarity as ssim

import os

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# Load model outside main function
vgg_model = tf.keras.applications.VGG16(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)


@st.cache
def load_model(model_filename):
    return joblib.load(model_filename)


@st.cache
def preprocess_image(image: np.ndarray) -> np.ndarray:
    img_resized = cv2.resize(image, (224, 224))
    return preprocess_input(img_resized)


def ssi_image_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    try:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
        similarity_score, _ = ssim(img1_gray, img2_gray, full=True)
        return similarity_score
    except Exception as e:
        st.error(f"Error calculating the similarity score: {e}")
        return 0


def classify_image(img_prediction: float) -> str:
    if img_prediction > 0.5:
        return "Pump Impeller/Washer Image is NOT DEFECTIVE(Ok_front)"
    else:
        return "Pump Impeller/Washer Image is DEFECTIVE(def_front)"


def main():
    st.header("Classifying Defects in Submersible Pump Impellers")
    st.subheader(
        "Categorizes submersible pump impellers into two primary groups: Defective or Non-Defective"
    )

    # Load the reference image
    reference_image_path = "cast_def_1.jpeg"
    reference_image_array = cv2.imread(reference_image_path, cv2.IMREAD_COLOR)
    uploaded_image = st.file_uploader(
        "Upload an image of the top view of a submersible pump impeller:",
        type=["jpg", "png", "jpeg"],
    )

    col1, col2 = st.columns(2)

    # Display the reference image in the first column
    with col1:
        st.write("Reference Image:")
        st.image(reference_image_array, use_column_width=True)

    if uploaded_image is not None:
        try:
            # Read the uploaded image
            uploaded_image_array = cv2.imdecode(
                np.frombuffer(uploaded_image.read(), np.uint8), cv2.IMREAD_COLOR
            )
        except Exception as e:
            st.error(f"Error reading the uploaded image: {e}")
            return

        similarity_score = ssi_image_similarity(
            uploaded_image_array, reference_image_array
        )

        if similarity_score >= 0.05:
            # Display the uploaded image in the second column
            with col2:
                st.write("Uploaded Image:")
                st.image(uploaded_image_array, use_column_width=True)

            # Preprocess the user input image
            preprocessed_image = preprocess_image(uploaded_image_array)

            # Extract features using VGG16
            input_img_feature = vgg_model.predict(
                np.expand_dims(preprocessed_image, axis=0)
            )
            input_img_features = input_img_feature.reshape(
                input_img_feature.shape[0], -1
            )

            # Load the saved SVC Model
            model_filename = "vgg_svc_model.joblib"
            model = load_model(model_filename)

            # Make predictions with probabilities
            prediction_probs = model.predict(input_img_features)[0]

            st.success(
                f"Uploaded image is similar to the reference image with Similarity Score of {similarity_score:.2f}"
            )

            # Display the prediction
            st.write("Classification Result:")
            result = classify_image(prediction_probs)
            st.write(result)

        else:
            with col2:
                st.write("Uploaded Image:")
            st.warning(
                f"The uploaded image has low similarity to the reference image. Similarity Score: {similarity_score:.2f}"
            )


if __name__ == "__main__":
    main()
