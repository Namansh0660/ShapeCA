import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pandas as pd

# =========================
# PAGE CONFIG
# =========================
st.set_page_config(
    page_title="Shape & Contour Analyzer",
    page_icon="📐",
    layout="wide"
)

st.title("ShapeCA: Shape & Contour Analyzer")
st.markdown("All processing stages shown simultaneously for clear comparison")

# =========================
# SIDEBAR
# =========================
with st.sidebar:
    # Custom HTML Student Card
    st.markdown("""
    <div class="student-card">
        <b>👤 Submitted By:</b><br>
        <span style="font-size: 1.1em;">Namansh Singh Maurya</span><br>
        <span style="font-size: 0.9em; opacity: 0.8;">Reg: 22MIA1034</span><br>
        <span style="font-size: 0.9em; opacity: 0.8;">Course: CSE3089 - Computer Vision</span>
    </div>
    """, unsafe_allow_html=True)
    st.markdown("---")
st.sidebar.header("📁 Image Upload")

file = st.sidebar.file_uploader(
    "Upload an image",
    type=["png", "jpg", "jpeg"]
)

st.sidebar.markdown("---")
st.sidebar.header("⚙️ Analysis Parameters")

preprocessing_method = st.sidebar.selectbox(
    "Preprocessing Method",
    ["Canny Edge", "Otsu Threshold", "Simple Threshold", "Adaptive Threshold"]
)

blur_kernel = st.sidebar.slider("Blur Kernel Size", 1, 15, 5, step=2)
min_area = st.sidebar.slider("Minimum Area (pixels)", 50, 2000, 300)
max_area = st.sidebar.slider("Maximum Area (pixels)", 5000, 50000, 30000)
approx_epsilon = st.sidebar.slider("Contour Approximation", 0.01, 0.08, 0.02, 0.01)

# =========================
# IMAGE RESIZE
# =========================
def resize_for_display(pil_img, max_width=400, max_height=300):
    w, h = pil_img.size
    scale = min(max_width / w, max_height / h, 1)
    return pil_img.resize(
        (int(w * scale), int(h * scale)),
        Image.Resampling.LANCZOS
    )

# =========================
# SHAPE DETECTION
# =========================
def detect_shape(cnt, eps):
    peri = cv2.arcLength(cnt, True)
    area = cv2.contourArea(cnt)

    if peri == 0 or area == 0:
        return "Unknown", 0, 0, 0

    approx = cv2.approxPolyDP(cnt, eps * peri, True)
    v = len(approx)

    x, y, w, h = cv2.boundingRect(approx)
    ar = w / h if h else 0
    circ = (4 * np.pi * area) / (peri ** 2)

    if v == 3:
        shape = "Triangle"
    elif v == 4:
        shape = "Square" if 0.9 <= ar <= 1.1 else "Rectangle"
    elif v == 5:
        shape = "Pentagon"
    elif v == 6:
        shape = "Hexagon"
    elif v == 7:
        shape = "Heptagon"
    elif v >= 8:
        if circ >= 0.85 and 0.9 <= ar <= 1.1:
            shape = "Circle"
        elif circ >= 0.65 and 0.85 <= ar <= 1.15:
            shape = "Probable Circle"
        else:
            shape = "Irregular"
    else:
        shape = "Unknown"

    return shape, v, round(circ, 3), round(ar, 3)

# =========================
# PREPROCESS IMAGE
# =========================
def preprocess(img):
    """
    Ensures image is:
    - Single channel (grayscale)
    - uint8
    - Compatible with all OpenCV thresholding & contour ops
    """

    # Handle RGBA, RGB, Grayscale safely
    if len(img.shape) == 3:
        if img.shape[2] == 4:  # RGBA
            gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)
        elif img.shape[2] == 3:  # RGB
            gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        else:
            gray = img[:, :, 0]
    else:
        gray = img.copy()

    # Ensure uint8
    if gray.dtype != np.uint8:
        gray = gray.astype(np.uint8)

    # Blur (noise reduction)
    blur = cv2.GaussianBlur(gray, (blur_kernel, blur_kernel), 0)

    # Thresholding
    if preprocessing_method == "Canny Edge":
        th = cv2.Canny(blur, 50, 150)
        th = cv2.dilate(th, np.ones((3, 3), np.uint8), 1)

    elif preprocessing_method == "Otsu Threshold":
        _, th = cv2.threshold(
            blur, 0, 255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    elif preprocessing_method == "Simple Threshold":
        _, th = cv2.threshold(
            blur, 127, 255,
            cv2.THRESH_BINARY_INV
        )

    else:  # Adaptive Threshold
        th = cv2.adaptiveThreshold(
            blur, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV,
            11, 2
        )

    return th


# =========================
# PROCESS IMAGE
# =========================
def process(image):
    img = np.array(image)
    thresh = preprocess(img)

    contours, _ = cv2.findContours(
        thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )

    output = img.copy()
    contour_img = img.copy()
    data = []
    idx = 1

    for cnt in contours:
        area = cv2.contourArea(cnt)
        if min_area < area < max_area:

            shape, v, circ, ar = detect_shape(cnt, approx_epsilon)

            # Merge Probable Circle → Circle
            if shape == "Probable Circle":
                shape = "Circle"

            peri = cv2.arcLength(cnt, True)
            M = cv2.moments(cnt)
            cx = int(M["m10"] / M["m00"]) if M["m00"] else 0
            cy = int(M["m01"] / M["m00"]) if M["m00"] else 0

            cv2.drawContours(output, [cnt], -1, (0, 255, 0), 2)
            cv2.putText(output, shape, (cx - 30, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

            cv2.drawContours(contour_img, [cnt], -1, (255, 0, 0), 2)

            data.append({
                "ID": idx,
                "Shape": shape,
                "Vertices": v,
                "Area": round(area, 2),
                "Perimeter": round(peri, 2),
                "Circularity": circ,
                "Aspect Ratio": ar
            })
            idx += 1

    return output, contour_img, thresh, pd.DataFrame(data)

# =========================
# MAIN LOGIC
# =========================
if file:
    image = Image.open(file)

    with st.spinner("🔍 Analyzing image..."):
        output_img, contour_img, thresh_img, df = process(image)

    st.success(f"Detected {len(df)} shapes")

    # =========================
    # IMAGE STAGES
    # =========================
    st.markdown("## 🖼 Image Processing Stages")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.subheader("Original")
        st.image(resize_for_display(image))

    with col2:
        st.subheader("Preprocessed")
        st.image(resize_for_display(Image.fromarray(thresh_img)))

    with col3:
        st.subheader("Contours")
        st.image(resize_for_display(Image.fromarray(contour_img)))

    with col4:
        st.subheader("Classified")
        st.image(resize_for_display(Image.fromarray(output_img)))

    # =========================
    # ADVANCED SHAPE ANALYSIS
    # =========================
    st.markdown("---")
    st.markdown("## 📊 Shape Analysis Dashboard")

    if not df.empty:

        # =========================
        # KEY METRICS (TOP ROW)
        # =========================
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Total Shapes", len(df))
        c2.metric("Unique Shape Types", df["Shape"].nunique())
        c3.metric("Total Area", f"{df['Area'].sum():,.0f}")
        c4.metric("Avg Circularity", f"{df['Circularity'].mean():.2f}")
        c5.metric("Avg Aspect Ratio", f"{df['Aspect Ratio'].mean():.2f}")

        st.markdown("### 🔢 Shape Count Distribution")
        shape_count_df = df["Shape"].value_counts().reset_index()
        shape_count_df.columns = ["Shape Type", "Count"]
        st.dataframe(shape_count_df, use_container_width=True, hide_index=True)

        st.markdown("### 📋 Shape Properties")
        st.dataframe(df, use_container_width=True)

        # =========================
        # AREA & PERIMETER ANALYSIS
        # =========================
        st.markdown("### 📐 Size & Geometry Analysis")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Area per Shape (px²)**")
            st.bar_chart(df.set_index("ID")["Area"])

        with col2:
            st.markdown("**Perimeter per Shape (px)**")
            st.bar_chart(df.set_index("ID")["Perimeter"])

        # =========================
        # SHAPE TYPE STATISTICS
        # =========================
        st.markdown("### 📊 Shape-wise Statistical Summary")

        stats_df = (
            df.groupby("Shape")
            .agg(
                Count=("ID", "count"),
                Avg_Area=("Area", "mean"),
                Avg_Perimeter=("Perimeter", "mean"),
                Avg_Circularity=("Circularity", "mean"),
                Avg_Aspect_Ratio=("Aspect Ratio", "mean"),
            )
            .round(2)
            .reset_index()
        )

        st.dataframe(stats_df, use_container_width=True)

        # =========================
        # INDIVIDUAL SHAPE INSPECTOR
        # =========================
        st.markdown("### 🔍 Individual Shape Inspector")

        selected_id = st.selectbox("Select Shape ID", df["ID"])

        shape_row = df[df["ID"] == selected_id].iloc[0]

        col1, col2, col3 = st.columns(3)

        col1.metric("Shape Type", shape_row["Shape"])
        col2.metric("Area (px²)", f"{shape_row['Area']:.0f}")
        col3.metric("Perimeter (px)", f"{shape_row['Perimeter']:.0f}")

        col1.metric("Vertices", int(shape_row["Vertices"]))
        col2.metric("Circularity", f"{shape_row['Circularity']:.2f}")
        col3.metric("Aspect Ratio", f"{shape_row['Aspect Ratio']:.2f}")

        # =========================
        # EXPORT SECTION
        # =========================
        st.markdown("### 💾 Export Analysis Results")

        st.download_button(
            "⬇️ Download Full Shape Analysis (CSV)",
            df.to_csv(index=False),
            "shape_analysis_full.csv",
            "text/csv"
        )

    else:
        st.warning("No shapes detected. Adjust parameters and try again.")


else:
    st.info("👈 Upload an image from the sidebar to begin analysis")

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown(
    "<center><b>Shape & Contour Analyzer</b> | Made by Namansh</center>",
    unsafe_allow_html=True
)
