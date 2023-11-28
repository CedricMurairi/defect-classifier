import streamlit as st
from PIL import Image
import joblib
from keras.applications.vgg16 import VGG16, preprocess_input
import numpy as np
import cv2
from skimage.metrics import structural_similarity as ssim
from skimage.feature import hog
from skimage import exposure


def sift_image_similarity(img1, img2):
    sift = cv2.SIFT_create()
    kp1, des1 = sift.detectAndCompute(img1, None)
    kp2, des2 = sift.detectAndCompute(img2, None)
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    good_matches = [m for m, n in matches if m.distance < 0.75 * n.distance]
    similarity_score = len(good_matches) / max(len(kp1), len(kp2))
    return similarity_score


def hog_image_similarity(img1, img2):
    img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    _, hog_img1 = hog(img1_gray, visualize=True)
    _, hog_img2 = hog(img2_gray, visualize=True)
    hog_img1 = exposure.rescale_intensity(hog_img1, in_range=(0, 10))
    hog_img2 = exposure.rescale_intensity(hog_img2, in_range=(0, 10))
    hog_img1 = (hog_img1 * 255).astype(np.uint8)
    hog_img2 = (hog_img2 * 255).astype(np.uint8)
    similarity_score, _ = ssim(
        hog_img1, hog_img2, full=True, data_range=hog_img1.max() - hog_img1.min())
    return similarity_score


def is_similar(img1, img2):
    try:
        sift_score = sift_image_similarity(img1, img2)
        hog_score = hog_image_similarity(img1, img2)
        combined_score = 0.6 * sift_score + 0.4 * hog_score
        similarity_threshold = 0.6
        return combined_score >= similarity_threshold
    except Exception as e:
        st.error(e)
        return False



def preprocess_image_vgg(image):
    img_resized = cv2.resize(image, (224, 224))
    img_resized = cv2.cvtColor(img_resized, cv2.COLOR_RGB2BGR)
    img_resized = preprocess_input(img_resized)
    img_resized = img_resized / 255
    return img_resized


def extract_features_vgg(model, image):
    return model.predict(np.expand_dims(image, axis=0)).reshape(-1)


def classify_defect(image_features, model):
    # Ensure the input array is 2D
    image_features_2d = image_features.reshape(1, -1)

    prediction = model.predict(image_features_2d)[0]

    return 'Washer Image is NOT DEFECTIVE(Ok_front)' if prediction > 0.5 else 'Washer Image is DEFECTIVE(def_front)'


def main():
    st.title("Defect Classification App")

    reference_image_path = "cast_ok_0_1155.jpeg"
    reference_image = cv2.imread(reference_image_path)

    uploaded_file = st.file_uploader(
        "Upload an image of top view of a submersible pump impeller:", type=["jpg", "png", "jpeg"])

    col1, col2 = st.columns(2)

    with col1:
        st.write("Reference Image:")
        st.image(reference_image, use_column_width=True)

    if uploaded_file is not None:
        uploaded_image = Image.open(uploaded_file)
        img1 = cv2.cvtColor(np.array(uploaded_image), cv2.COLOR_RGB2BGR)

        # Image Similarity Check
        if not is_similar(img1, reference_image):
            st.warning(
                "Uploaded image is not inherently similar to the reference image.")
        else:
            # Image Preprocessing and Classification using VGG16
            img1_processed = preprocess_image_vgg(img1)
            vgg_model = VGG16(weights='imagenet',
                              include_top=False, input_shape=(224, 224, 3))
            img1_features = extract_features_vgg(vgg_model, img1_processed)

            model_filename = "vgg_svc_model.joblib"
            vgg16_model = joblib.load(model_filename)

            result = classify_defect(img1_features, vgg16_model)

            with col2:
                st.write("Uploaded Image:")
                st.image(uploaded_image, use_column_width=True)

            st.success(f"Prediction: {result}")


if __name__ == "__main__":
    main()
