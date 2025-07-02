from fastapi import FastAPI, UploadFile, File, Form, Query
from fastapi.responses import StreamingResponse, JSONResponse, FileResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
import cv2
import numpy as np
import face_recognition
from src.face_detector import YOLOv5
from src.FaceAntiSpoofing import AntiSpoof
import os
import json
from typing import Optional
from io import BytesIO
from urllib.parse import unquote

app = FastAPI(
    title="SCC | Face Recognition",
    description="API для распознавания лиц и антиспуфинга",
    version="1.0"
)

# Разрешаем CORS для фронта
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Загрузка переводов
with open("static/lang.json", "r", encoding="utf-8") as f:
    translations = json.load(f)

# Загрузка моделей
face_detector = YOLOv5('saved_models/yolov5s-face.onnx')
anti_spoof = AntiSpoof('saved_models/AntiSpoofing_bin_1.5_128.onnx')

# Загрузка известных лиц
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

def get_eye_distance(img):
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    landmarks_list = face_recognition.face_landmarks(rgb_img)
    if not landmarks_list:
        return None
    landmarks = landmarks_list[0]
    if "left_eye" in landmarks and "right_eye" in landmarks:
        left_eye = np.mean(landmarks["left_eye"], axis=0)
        right_eye = np.mean(landmarks["right_eye"], axis=0)
        return np.linalg.norm(np.array(left_eye) - np.array(right_eye))
    return None

def process_frame(frame, threshold=0.5, eye_dist_thresh=40):
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    bbox = face_detector([img_rgb])[0]
    if bbox.shape[0] == 0:
        return frame
    bbox = bbox.flatten()[:4].astype(int)
    face_crop = increased_crop(img_rgb, bbox)

    # Сначала считаем расстояние между глазами
    eye_dist = get_eye_distance(face_crop)

    if eye_dist is not None and eye_dist < eye_dist_thresh:
        label_text = f"TOO CLOSE (Eyes: {eye_dist:.1f})"
        color = (0, 165, 255)
    else:
        # Только если лицо не слишком близко — антиспуфинг
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

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    processed = process_frame(frame)
    _, img_encoded = cv2.imencode('.jpg', processed)
    return StreamingResponse(BytesIO(img_encoded.tobytes()), media_type="image/jpeg")

@app.post("/recognize/")
async def recognize(file: UploadFile = File(...)):
    contents = await file.read()
    npimg = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
    name = recognize_face(frame)
    return JSONResponse({"name": name})

@app.get("/video/")
async def video_stream(
    source: str = Query("local"),
    ip_url: Optional[str] = Query(None)
):
    import time
    import requests

    cap = None
    is_ip = False
    error_msg = None

    if source == "local":
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        is_ip = False
        if not cap.isOpened():
            error_msg = "Локальная камера не найдена или занята другим приложением."
    else:
        if not ip_url:
            return JSONResponse({"error": "IP camera URL required"}, status_code=400)
        # Удаляем пробелы и плюсы, декодируем url
        ip_url_clean = unquote(ip_url).strip().replace('+', '')
        print(f"Попытка подключения к IP-камере: {ip_url_clean}")
        cap = cv2.VideoCapture(ip_url_clean)
        is_ip = True
        if not cap.isOpened():
            # Попробуем получить кадр через HTTP-запрос (например, /shot.jpg)
            cap = None
            try:
                url = ip_url_clean
                if url.endswith("/video"):
                    url = url.replace("/video", "/shot.jpg")
                elif not url.endswith(".jpg"):
                    url = url.rstrip("/") + "/shot.jpg"
                print(f"Попытка получить кадр по HTTP: {url}")
                resp = requests.get(url, timeout=2)
                arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                test_frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if test_frame is None:
                    error_msg = "IP-камера не отвечает или не поддерживает поток/кадры."
            except Exception as e:
                error_msg = f"Ошибка подключения к IP-камере: {e}"

    if error_msg:
        # Вернуть ошибку в виде изображения с текстом
        def error_gen():
            img = np.zeros((360, 480, 3), dtype=np.uint8)
            cv2.putText(img, "Ошибка:", (20, 100), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0,0,255), 3)
            y = 160
            for line in error_msg.split('\n'):
                cv2.putText(img, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                y += 40
            _, buffer = cv2.imencode('.jpg', img)
            while True:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
                time.sleep(1)
        return StreamingResponse(error_gen(), media_type='multipart/x-mixed-replace; boundary=frame')

    def gen():
        last_frame = None
        while True:
            frame = None
            if not is_ip or (cap and cap.isOpened()):
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
            else:
                try:
                    url = unquote(ip_url).strip().replace('+', '')
                    if url.endswith("/video"):
                        url = url.replace("/video", "/shot.jpg")
                    elif not url.endswith(".jpg"):
                        url = url.rstrip("/") + "/shot.jpg"
                    resp = requests.get(url, timeout=2)
                    arr = np.asarray(bytearray(resp.content), dtype=np.uint8)
                    frame = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                    if frame is None:
                        break
                except Exception:
                    break
            if frame is None:
                break
            frame = cv2.resize(frame, (480, 360), interpolation=cv2.INTER_LINEAR)
            processed = process_frame(frame)
            _, buffer = cv2.imencode('.jpg', processed)
            last_frame = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + last_frame + b'\r\n')
            time.sleep(0.04)  # ~25 fps
        if cap:
            cap.release()

    return StreamingResponse(gen(), media_type='multipart/x-mixed-replace; boundary=frame')

# Подключаем папку static для логотипа и других файлов
app.mount("/static", StaticFiles(directory="static"), name="static")

# Отдаём frontend при обращении к корню
@app.get("/")
async def root():
    return FileResponse("frontend/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "video_predict:app",
        host="192.168.1.10",
        port=8000,
        reload=False,
        ssl_keyfile="OpenSSL/key.pem",
        ssl_certfile="OpenSSL/cert.pem"
    )
