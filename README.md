# ShapeCA

An interactive Streamlit application for detecting, classifying, and analyzing geometric shapes in images using OpenCV contour analysis.

---

## Project Overview

This project detects geometric shapes from images, counts objects, and computes key properties such as area, perimeter, circularity, and aspect ratio.  
It provides a real-time dashboard where all image processing stages are visualized simultaneously.

---

## Features

- Image upload via sidebar
- Simultaneous display of:
  - Original image
  - Preprocessed image
  - Contour detection
  - Classified shapes
- Adjustable preprocessing parameters
- Detection and classification of geometric shapes
- Shape-wise analysis and statistics
- Export shape data as CSV

---

## Supported Shapes

- Triangle
- Square
- Rectangle
- Pentagon
- Hexagon
- Heptagon
- Circle
- Irregular shapes

---

## Tech Stack

- Python
- Streamlit
- OpenCV
- NumPy
- Pandas
- Pillow (PIL)

---

## How to Run

```bash
pip install -r requirements.txt
streamlit run app.py
```

# Learning Outcomes
- Understanding contour-based shape detection
- Feature extraction using geometric properties
- Practical use of OpenCV
- Building interactive dashboards with Streamlit

# Author
- Namansh Singh Maurya
- Computer Vision Project
