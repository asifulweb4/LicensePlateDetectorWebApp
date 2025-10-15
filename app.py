import os
os.system("apt-get update && apt-get install -y libgl1 libglib2.0-0")

import streamlit as st
from PIL import Image
import numpy as np
import os
import cv2
from ultralytics import YOLO
from datetime import datetime
import pandas as pd
import pytesseract
import platform

# মনে রাখবেন: এই utils.py ফাইলটি আপনার প্রজেক্টে থাকতে হবে 
# এবং তাতে প্রয়োজনীয় function গুলো (ensure_dirs, save_cropped_image, ocr_image, init_db, insert_detection, clear_captures_and_db) সংজ্ঞায়িত করা থাকতে হবে।
from utils import ensure_dirs, save_cropped_image, ocr_image, init_db, insert_detection, clear_captures_and_db

# --- Auto-detect paths ---
def get_paths():
    if platform.system() == "Windows":
        # আপনার Windows Path এখানে চেক করুন
        tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"
        model_path = r"E:\LicensePlate_webApp\model\best.pt"
        db_path = r"E:\LicensePlate_webApp\data\plates.db"
        capture_dir = r"E:\LicensePlate_webApp\static\captures"
    else:
        # Linux/Mac paths
        tesseract_cmd = "/usr/bin/tesseract"
        model_path = "model/best.pt"
        db_path = "data/plates.db"
        capture_dir = "static/captures"
    return tesseract_cmd, model_path, db_path, capture_dir

TESSERACT_CMD, MODEL_PATH, DB_PATH, CAPTURE_DIR = get_paths()
pytesseract.pytesseract.tesseract_cmd = TESSERACT_CMD

# --- Streamlit Setup ---
st.set_page_config(page_title="License Plate Detector", page_icon="🚘", layout="wide")
st.markdown("<h1 style='text-align:center;color:#008080'>🚗 Number Plate Detector Using Deep Learning Approach</h1>", unsafe_allow_html=True)

# Sidebar Settings
st.sidebar.header("⚙️ Settings")
conf = st.sidebar.slider("Detection Confidence", 0.1, 0.9, 0.25, 0.05)
# Tesseract PSM (Page Segmentation Mode) 7/8 সাধারণত একক বস্তু সনাক্তকরণের জন্য ভালো
psm = st.sidebar.selectbox("Tesseract PSM", [3, 6, 7, 8], index=2)
lang = st.sidebar.text_input("OCR Language", value="eng")
st.sidebar.markdown("---")
st.sidebar.write("👨‍💻 Developed by Asiful Islam")
st.sidebar.markdown("---")

# --- Initialize DB & Folders ---
conn = init_db(DB_PATH)
ensure_dirs(CAPTURE_DIR)

# --- Session State ---
if "input_image" not in st.session_state:
    st.session_state.input_image = None
if "results_list" not in st.session_state:
    st.session_state.results_list = []
# Clear Button FIX: widget state reset করার জন্য key
if "uploader_key" not in st.session_state:
    st.session_state.uploader_key = 0

# Clear function
def clear_all():
    st.session_state.input_image = None
    st.session_state.results_list = []
    # FIX: Key পরিবর্তন করে uploader widget reset করা
    st.session_state.uploader_key += 1 
    clear_captures_and_db(conn, CAPTURE_DIR)

if st.sidebar.button("🗑️ Clear All Captures & DB"):
    clear_all()
    st.sidebar.success("All captures and database records cleared!")

# --- Load YOLO ---
@st.cache_resource
def load_model(path):
    return YOLO(path)

model = load_model(MODEL_PATH)

# --- Upload / Camera ---
col1, col2 = st.columns(2)
with col1:
    # Key ব্যবহার করে file uploader reset করা হবে
    uploaded = st.file_uploader("📁 Upload an image", type=['jpg','jpeg','png'], 
                                key=f"uploader_{st.session_state.uploader_key}") 
with col2:
    # Key ব্যবহার করে camera input reset করা হবে
    cam = st.camera_input("📸 Capture from camera", 
                          key=f"camera_{st.session_state.uploader_key}")

if uploaded:
    st.session_state.input_image = Image.open(uploaded).convert("RGB")
elif cam:
    st.session_state.input_image = Image.open(cam).convert("RGB")

# Display input
if st.session_state.input_image:
    st.image(st.session_state.input_image, caption="Selected Image", use_column_width=True)
    detect_btn = st.button("🚀 Detect License Plate")
else:
    st.info("Please upload an image or capture using your camera.")
    detect_btn = False

# --- Detection ---
if detect_btn and st.session_state.input_image:
    with st.spinner("Running YOLO detection..."):
        input_image = st.session_state.input_image
        img_np = np.array(input_image)
        results = model.predict(source=img_np, imgsz=640, conf=conf, device="cpu", verbose=False)
        res = results[0]

        xyxy = res.boxes.xyxy.cpu().numpy() if hasattr(res.boxes, "xyxy") else []
        confs = res.boxes.conf.cpu().numpy() if hasattr(res.boxes, "conf") else []

        if len(xyxy) == 0:
            st.warning("No plates detected! Try lowering confidence.")
        else:
            st.success(f"✅ Detected {len(xyxy)} plate(s)")
            results_list = []

            img_cv = np.array(input_image)
            img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

            for i, box in enumerate(xyxy):
                x1, y1, x2, y2 = map(int, box)
                conf_score = float(confs[i])

                crop = input_image.crop((x1, y1, x2, y2))
                plate_text = ocr_image(crop, psm=psm, lang=lang)

                # Draw bounding box + OCR text (Final Fixes)
                
                # 1. Text Content: Plate Text এবং Score উভয়ই একত্রিত করা 
            if plate_text:
                # Plate text পাওয়া গেলে, text এবং score উভয়ই দেখাবে (যেমন: AAAAA (0.95))
                    display_text = f"{plate_text} ({conf_score:.2f})"
            else:
                # Plate text না পাওয়া গেলে, শুধুমাত্র score দেখাবে (যেমন: (0.88))
                display_text = f"({conf_score:.2f})" 
                
                # 2. Bounding Box আঁকা
                cv2.rectangle(img_cv, (x1, y1), (x2, y2), (0,255,0), 2)

                # 3. Text আঁকা (y1-15 ব্যবহার করা হলো text-টিকে Bounding Box এর উপরে দেখানোর জন্য)
                cv2.putText(img_cv, display_text, (x1, y1 - 15),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                
                saved_path = save_cropped_image(crop, CAPTURE_DIR, "plate")
                insert_detection(conn, saved_path, plate_text, conf_score)

                results_list.append({"img": saved_path, "text": plate_text, "conf": conf_score,
                                     "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})

            st.session_state.results_list = results_list

            img_with_boxes = Image.fromarray(cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB))
            st.image(img_with_boxes, caption="Detected Plates with Bounding Boxes", use_column_width=True)

            # Cropped Plates
            st.subheader("📋 Cropped Plate Images")
            cols = st.columns(3)
            for i, r in enumerate(results_list):
                with cols[i % 3]:
                    # Cropped image এর নিচে plate text/N/A এবং score দেখানো হচ্ছে
                    caption_text = f"{r['text'] if r['text'] else 'N/A'} ({r['conf']:.2f})"
                    st.image(r['img'], caption=caption_text, use_column_width=True)

# --- Recent + CSV ---
st.markdown("---")
st.header("🕘 Recent Detections")
c = conn.cursor()
c.execute("SELECT id, image_path, plate_text, confidence, timestamp FROM detections ORDER BY id DESC")
rows = c.fetchall()

if rows:
    df = pd.DataFrame(rows, columns=["ID","Image Path","Plate Text","Confidence","Timestamp"])
    st.dataframe(df, use_container_width=True)
    st.download_button("📥 Download CSV", df.to_csv(index=False).encode('utf-8'),
                        file_name="recent_detections.csv", mime="text/csv")
else:
    st.info("No previous detections found.")

st.markdown('<p style="text-align:center;color:gray;">Made with ❤️ using Streamlit, YOLOv8 & Tesseract OCR</p>', unsafe_allow_html=True)