# 🌊 AI-Powered Ocean Plastics Waste Detection using YOLOv8

> **Real-time object detection and classification for floating marine debris across four waste categories.**

[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue?style=flat)](https://github.com/ultralytics/ultralytics)
[![Python](https://img.shields.io/badge/Python-3.9%2B-yellow?style=flat&logo=python)](https://python.org)
[![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit)](https://streamlit.io)
[![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org)

---

## 📌 Overview

Plastic pollution in oceans poses severe threats to marine ecosystems. This project delivers an automated Computer Vision solution powered by **YOLOv8 (You Only Look Once)**, fine-tuned on marine waste datasets to accurately detect and classify floating trash in ocean imagery.

### 🎯 Detected Waste Classes
1. 🍶 **Glass**
2. 🥫 **Metal**
3. 🥤 **Plastic**
4. 🗑️ **General Trash**

---

## ✨ Features

- ⚡ **High-Accuracy Detection:** Custom fine-tuned weights (`best.pt`) optimized for marine and underwater visibility variations.
- 🖥️ **Interactive Streamlit Interface:** User-friendly web dashboard allowing instant image uploads and detection preview.
- 🔍 **Visual Bounding Boxes:** Dynamic drawing of class names and confidence percentages directly on uploaded images.
- 📊 **Confidence Tuning:** Configured with a 25% minimum threshold to balance precision and recall.

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/roshan749997/AI-powered-tool-for-detecting-waste-materials-in-the-ocean-using-YOLOv8.git
cd AI-powered-tool-for-detecting-waste-materials-in-the-ocean-using-YOLOv8
```

### 2. Set Up Virtual Environment & Dependencies
```bash
python -m venv venv
source venv/bin/activate  # On Windows: .\venv\Scripts\activate
pip install ultralytics streamlit pillow torch torchvision
```

### 3. Launch the Application
```bash
streamlit run main_file.py
```
Open your browser at `http://localhost:8501` to use the web application.

---

## 📁 Repository Structure
* `main_file.py` — Streamlit interactive web application.
* `best.pt` — Trained YOLOv8 model weights.
* `Ocean_Plastic_Waste_Detection.ipynb` — Training, validation, and evaluation notebook.
* `Ocean Plastic Waste Detection Roshan_Chaudhari (2).pptx` — Project overview and architecture presentation.
