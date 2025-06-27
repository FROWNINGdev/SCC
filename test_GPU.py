import onnxruntime as ort

# Подставь путь к твоей ONNX-модели
model_path = "saved_models/yolov5s-face.onnx"  # замени на свой путь

# Создаем сессию с CUDA
session = ort.InferenceSession(model_path, providers=["CUDAExecutionProvider"])

print("✅ Провайдеры ONNX:", session.get_providers())
