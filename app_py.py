import streamlit as st
import cv2
import numpy as np  # Import NumPy
import pywt
from scipy.stats import skew, kurtosis
import joblib

# Load the trained model
model = joblib.load("random_forest_model.joblib")  # Updated file extension to .joblib

# Function to calculate variance, skewness, and kurtosis
def calculate_stats(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    coeffs = pywt.wavedec2(img_gray, 'haar', level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs
    flat_coeffs = cA2.flatten()
    variance = flat_coeffs.var()
    skewness = skew(flat_coeffs)
    kurt = kurtosis(flat_coeffs)
    return variance, skewness, kurt

# Streamlit UI
st.title("Fake Currency Classifier")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = cv2.imdecode(np.frombuffer(uploaded_file.read(), np.uint8), 1)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Classify'):
        stats = calculate_stats(image)
        result = model.predict([stats])
        if result[0] == 0:
            st.write("The uploaded note is classified as: Real")
        else:
            st.write("The uploaded note is classified as: Fake")
