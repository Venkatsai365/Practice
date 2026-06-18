import streamlit as st
import cv2
import numpy as np
import pywt
from scipy.stats import skew, kurtosis
import joblib

# ---------------- PAGE CONFIG ----------------
st.set_page_config(
    page_title="Fake Currency Detection",
    page_icon="💵",
    layout="wide"
)

# ---------------- CUSTOM CSS ----------------
st.markdown("""
<style>
.main-title{
    text-align:center;
    color:#1E88E5;
    font-size:50px;
    font-weight:bold;
}
.sub-title{
    text-align:center;
    color:gray;
    font-size:20px;
    margin-bottom:30px;
}
</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ----------------
model = joblib.load("random_forest_model.joblib")

# ---------------- FEATURE EXTRACTION ----------------
def calculate_stats(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    coeffs = pywt.wavedec2(img_gray, 'haar', level=2)
    cA2, (cH2, cV2, cD2), (cH1, cV1, cD1) = coeffs

    flat_coeffs = cA2.flatten()

    variance = flat_coeffs.var()
    skewness = skew(flat_coeffs)
    kurt = kurtosis(flat_coeffs)

    return variance, skewness, kurt

# ---------------- SIDEBAR ----------------
with st.sidebar:
    st.title("💵 Fake Currency Detector")

    st.info("""
    This application uses Machine Learning
    and Wavelet Transform features to
    classify currency notes as Real or Fake.
    """)

    st.markdown("---")

    st.subheader("How to Use")

    st.write("1. Upload currency image")
    st.write("2. Click Classify")
    st.write("3. View prediction result")

    st.markdown("---")

    st.success("Developed using Streamlit")

# ---------------- HEADER ----------------
st.markdown(
    '<p class="main-title">💵 Fake Currency Detection System</p>',
    unsafe_allow_html=True
)

st.markdown(
    '<p class="sub-title">Upload a currency note image and detect whether it is Real or Fake using Machine Learning.</p>',
    unsafe_allow_html=True
)

# ---------------- FILE UPLOAD ----------------
uploaded_file = st.file_uploader(
    "Upload Currency Note Image",
    type=["jpg", "jpeg", "png"]
)

# ---------------- IMAGE DISPLAY ----------------
if uploaded_file is not None:

    image = cv2.imdecode(
        np.frombuffer(uploaded_file.read(), np.uint8),
        1
    )

    col1, col2 = st.columns(2)

    with col1:
        st.image(
            cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            caption="Uploaded Image",
            width=400
        )

    with col2:

        st.subheader("Image Details")

        st.write(f"Height : {image.shape[0]} px")
        st.write(f"Width : {image.shape[1]} px")
        st.write(f"Channels : {image.shape[2]}")

        if st.button("🔍 Classify Currency"):

            variance, skewness, kurt = calculate_stats(image)

            result = model.predict(
                [[variance, skewness, kurt]]
            )

            st.write("Raw Prediction:", result[0])

            if result[0] == 1:
                st.success("✅ Currency Note is REAL")
            else:
                st.error("❌ Currency Note is FAKE")

            st.subheader("Extracted Features")

            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Variance",
                f"{variance:.2f}"
            )

            c2.metric(
                "Skewness",
                f"{skewness:.2f}"
            )

            c3.metric(
                "Kurtosis",
                f"{kurt:.2f}"
            )

# ---------------- FOOTER ----------------
st.markdown("---")

st.markdown(
    """
    <center>
    Built with ❤️ using Streamlit, OpenCV,
    PyWavelets and Machine Learning
    </center>
    """,
    unsafe_allow_html=True
)
