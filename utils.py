import os
from PIL import Image
import pytesseract
import sqlite3
from datetime import datetime
import shutil
import numpy as np
import cv2 # OpenCV library প্রয়োজন

# --- Helper functions (no changes needed) ---

def ensure_dirs(base_dir="static/captures"):
    os.makedirs(base_dir, exist_ok=True)
    return base_dir

def save_cropped_image(img_pil, base_dir="static/captures", prefix="plate"):
    ensure_dirs(base_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filename = f"{prefix}_{ts}.jpg"
    path = os.path.join(base_dir, filename)
    img_pil.save(path)
    return path

# --- OCR Image function (সংশোধিত অংশ) ---

def ocr_image(pil_img, psm=7, oem=3, lang='eng'):
    # PIL Image কে numpy array তে রূপান্তর
    img_cv = np.array(pil_img)
    # PIL RGB থেকে OpenCV BGR তে রূপান্তর
    img_cv = cv2.cvtColor(img_cv, cv2.COLOR_RGB2BGR)

    # 1. Grayscale
    gray_img = cv2.cvtColor(img_cv, cv2.COLOR_BGR2GRAY)
    
    # 2. Adaptive Thresholding (OCR এর জন্য Background/Foreground আলাদা করতে)
    # এই পদ্ধতিটি lighting conditions এর ভিন্নতা থাকা ছবিতে ভালো কাজ করে
    thresh_img = cv2.adaptiveThreshold(gray_img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    
    # 3. OCR Configuration সেট করা
    # psm এবং lang মান app.py থেকে আসে।
    config = f'--oem {oem} --psm {psm} -l {lang}'
    
    # 4. OCR চালানো
    try:
        # Pre-processed image (thresh_img) থেকে text extract করা
        plate_text = pytesseract.image_to_string(thresh_img, config=config)
        
        # 5. Text পরিষ্কার করা (শুধুমাত্র অক্ষর এবং সংখ্যা রাখা)
        cleaned_text = ''.join(filter(str.isalnum, plate_text)).upper()
        return cleaned_text
    except Exception:
        # OCR ব্যর্থ হলে empty string ফেরত দেওয়া
        return ""

# --- Database functions (no changes needed) ---

def init_db(db_path="data/plates.db"):
    os.makedirs(os.path.dirname(db_path), exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    c = conn.cursor()
    c.execute('''
        CREATE TABLE IF NOT EXISTS detections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            image_path TEXT,
            plate_text TEXT,
            confidence REAL,
            timestamp TEXT
        )
    ''')
    conn.commit()
    return conn

def insert_detection(conn, image_path, plate_text, confidence):
    c = conn.cursor()
    ts = datetime.now().isoformat()
    c.execute('''
        INSERT INTO detections (image_path, plate_text, confidence, timestamp)
        VALUES (?, ?, ?, ?)
    ''', (image_path, plate_text, confidence, ts))
    conn.commit()

def clear_captures_and_db(conn, capture_dir="static/captures"):
    # Delete all images
    if os.path.exists(capture_dir):
        shutil.rmtree(capture_dir)
    os.makedirs(capture_dir, exist_ok=True)
    # Clear database
    c = conn.cursor()
    c.execute("DELETE FROM detections")
    conn.commit()