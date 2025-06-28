import streamlit as st
import cv2
import numpy as np
import face_recognition
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import os
import json
import base64
from PIL import Image

# Настройка страницы
st.set_page_config(page_title="SCC | Face Recognition", layout="wide")

# Загрузка переводов
with open("lang.json", "r", encoding="utf-8") as f:
    translations = json.load(f)

# ===== CSS =====
st.markdown("""
    <style>
    .footer {
        position: fixed;
        bottom: 12px;
        width: 100%;
        text-align: center;
        font-size: 14px;
        color: gray;
    }
    .footer img {
        vertical-align: middle;
        margin-right: 6px;
        border-radius: 8px;
    }
    section[data-testid="stSidebar"] {
        background-color: transparent !important;
    }
    </style>
""", unsafe_allow_html=True)

# ===== Боковая панель (только язык и режим) =====
with st.sidebar:
    lang = st.selectbox(
        label="Language",
        options=["RU", "UZ", "EN"],
        index=0,
        key="lang_select",
        label_visibility="collapsed"
    )
    T = translations[lang]

    st.markdown(f"### {T['video_source']}")
    mode = st.radio(
        T["choose_mode"],
        [T["local_camera"], T["ip_camera"]],
        index=0
    )

# ===== Заголовок =====
st.title(T["app_title"])

# ===== Подвал (footer): логотип, автор, Telegram =====
logo_path = "static/image/logo.png"
if os.path.exists(logo_path):
    with open(logo_path, "rb") as f:
        logo_data = base64.b64encode(f.read()).decode()

    st.markdown(
        f"""
        <div class="footer">
            <a href="https://scc.uz/" target="_blank">
                <img src="data:image/png;base64,{logo_data}" width="26" alt="SCC Logo">
                SCC | https://scc.uz
            </a>
            <br>
            Автор: Абалкулов Амаль | <a href="https://t.me/FROWNINGnrx" target="_blank">@FROWNINGnrx</a>
        </div>
        """,
        unsafe_allow_html=True
    )
else:
    st.markdown(
        """
        <div class="footer">
            SCC | https://scc.uz<br>
            Автор: Абалкулов Амаль | <a href="https://t.me/FROWNINGnrx" target="_blank">@FROWNINGnrx</a>
        </div>
        """,
        unsafe_allow_html=True
    )

# ===== Загрузка моделей =====
face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

# ===== Загрузка известных лиц =====
known_face_encodings = []
known_face_names = []
student_db_path = "face_db"
for filename in os.listdir(student_db_path):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(student_db_path, filename)
        image = face_recognition.load_image_file(path)
        encodings = face_recognition.face_encodings(image)
        if encodings:
            known_face_encodings.append(encodings[0])
            name = os.path.splitext(filename)[0]
            known_face_names.append(name)

# ===== Кадрирование =====
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

# ===== Распознавание =====
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

# ===== Обработка кадра =====
def process_frame(frame, threshold=0.5):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bbox = face_detector([img_rgb])[0]
    if bbox.shape[0] == 0:
        return frame
    bbox = bbox.flatten()[:4].astype(int)
    face_crop = increased_crop(img_rgb, bbox)
    pred = anti_spoof([face_crop])[0]
    score = pred[0][0]
    label = np.argmax(pred)

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

    cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
    cv2.putText(frame, label_text, (bbox[0], bbox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# ===== Центр экрана =====
left, center, right = st.columns([1, 3, 1])
with center:
    st.subheader(mode)
    if mode == T["local_camera"]:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    else:
        ip_url = st.text_input(T["ip_prompt"], placeholder="http://IP:PORT/video")
        if not ip_url:
            st.warning(T["ip_warning"])
            st.stop()
        cap = cv2.VideoCapture(ip_url)

    FRAME_WINDOW = st.image([], width=640)

# ===== Обработка потока =====
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        st.warning("Не удалось захватить видео с источника")
        break
    frame = cv2.resize(frame, (640, 480), interpolation=cv2.INTER_LINEAR)
    processed = process_frame(frame)
    FRAME_WINDOW.image(processed, channels="BGR")

cap.release()
