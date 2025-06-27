# streamlit_app.py
import streamlit as st
import cv2
import numpy as np
import face_recognition
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import os
from PIL import Image

# Настройка страницы Streamlit
st.set_page_config(page_title="Программа для распознавания лица", layout="wide")
st.title("Программа для распознавания лица")

# Стилизация интерфейса и боковой панели
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 10px;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: gray;
    }
    section[data-testid="stSidebar"] {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# Боковая панель: выбор режима
with st.sidebar:
    if os.path.exists("static/logo.png"):
        st.image("static/logo.png", width=150)
    st.markdown("## Источник видеопотока")
    mode = st.radio("Выберите режим:", ["Локальная камера", "Подключение по IP"], index=0)

# Подвал с авторством
st.markdown("""
<div class="footer">Автор: Абалкулов Амаль | <img src="https://cdn-icons-png.flaticon.com/512/2111/2111646.png" width="16"> @FROWNINGnrx</div>
""", unsafe_allow_html=True)

# ================= ЗАГРУЗКА МОДЕЛЕЙ =================

# Детектор лиц: YOLOv5 (ONNX)
face_detector = YOLOv5('saved_models/yolov5s-face.onnx')

# Модель антиспуфинга: ONNX-версия
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

# ================= ЗАГРУЗКА ИЗВЕСТНЫХ ЛИЦ =================

known_face_encodings = []
known_face_names = []
student_db_path = "student_db"
for filename in os.listdir(student_db_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(student_db_path, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

# ================= КАДРИРОВАНИЕ ЛИЦА =================

def increased_crop(img, bbox, bbox_inc=1.5):
    real_h, real_w = img.shape[:2]
    x, y, w, h = bbox
    w, h = w - x, h - y
    l = max(w, h)
    xc, yc = x + w / 2, y + h / 2
    x, y = int(xc - l * bbox_inc / 2), int(yc - l * bbox_inc / 2)
    x1, y1 = max(0, x), max(0, y)
    x2, y2 = min(real_w, x + int(l * bbox_inc)), min(real_h, y + int(l * bbox_inc))
    crop = img[y1:y2, x1:x2, :]
    crop = cv2.copyMakeBorder(crop, y1 - y, int(l * bbox_inc - y2 + y), x1 - x, int(l * bbox_inc - x2 + x), cv2.BORDER_CONSTANT, value=[0, 0, 0])
    return crop

# ================= РАСПОЗНАВАНИЕ ЛИЦ =================

def recognize_face(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    faces_enc = face_recognition.face_encodings(rgb_img)
    if not faces_enc:
        return None
    face_encoding = faces_enc[0]
    matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    distances = face_recognition.face_distance(known_face_encodings, face_encoding)
    if any(matches):
        return known_face_names[np.argmin(distances)]
    return None

# ================= ОБРАБОТКА КАДРА =================

def process_frame(frame, threshold=0.5):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. Детекция лиц (YOLOv5)
    bbox = face_detector([img_rgb])[0]
    if bbox.shape[0] == 0:
        return frame
    bbox = bbox.flatten()[:4].astype(int)

    # 2. Кадрирование + антиспуфинг
    face_crop = increased_crop(img_rgb, bbox)
    pred = anti_spoof([face_crop])[0]
    score = pred[0][0]
    label = np.argmax(pred)

    # 3. Распознавание
    if label == 0 and score > threshold:
        name = recognize_face(face_crop) or "Unrecognized"
        label_text = f"REAL [{name}] ({score:.2f})"
        color = (0, 255, 0)
    elif label == 1:
        label_text = f"FAKE ({score:.2f})"
        color = (0, 0, 255)
    else:
        label_text = "UNKNOWN"
        color = (127, 127, 127)

    # Рисуем рамку и подпись
    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.putText(frame, label_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ================= ОТОБРАЖЕНИЕ ПОТОКА =================

# Центрированная область: заголовок и видео
left, center, right = st.columns([1, 3, 1])
with center:
    # Заголовок и камера
    if mode == "Локальная камера":
        st.subheader("Локальная камера")
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        st.subheader("Подключение по IP")
        ip_url = st.text_input("Введите URL IP-камеры:", placeholder="http://IP:PORT/video")
        if not ip_url:
            st.warning("Введите корректный IP-адрес камеры")
            st.stop()
        cap = cv2.VideoCapture(ip_url)

    # Отображение кадра
    FRAME_WINDOW = st.image([], width=640)

# Основной цикл обработки видео
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Не удалось захватить видео с источника")
        break

    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    processed = process_frame(frame)
    FRAME_WINDOW.image(processed, channels="BGR")

cap.release()