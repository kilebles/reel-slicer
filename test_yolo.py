import torch
import cv2
import os
import subprocess
import numpy as np
from ultralytics import YOLO

# 1. Инициализация
device = "0" if torch.cuda.is_available() else "cpu"
model = YOLO("yolov8m-pose.pt")

input_path = "data/output/segment_01_insight_score8.mp4"
output_dir = "data/cropped"
temp_output_path = os.path.join(output_dir, "temp_no_audio.mp4")
output_path = os.path.join(output_dir, "result_vertical_tripod.mp4")

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

cap = cv2.VideoCapture(input_path)
fps, width, height = int(cap.get(5)), int(cap.get(3)), int(cap.get(4))
target_w = int(height * 9 / 16)
if target_w % 2 != 0:
    target_w -= 1

out = cv2.VideoWriter(
    temp_output_path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (target_w, height)
)

# --- ПАРАМЕТРЫ "УМНОГО ШТАТИВА" ---
smooth_x = width // 2
is_moving = False  # Состояние: движется камера или стоит
trigger_threshold = 40  # Смещение в пикселях, которое ЗАСТАВИТ камеру проснуться
stop_threshold = (
    5  # Насколько точно нужно довести камеру до центра, чтобы она снова "заснула"
)
ease_speed = 0.12  # Скорость перестановки (быстрая и плавная)
# ---------------------------------

print("Запуск режима 'Умный штатив'...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    results = model(frame, verbose=False, conf=0.5, device=device)
    target_x = None

    if len(results[0].keypoints) > 0:
        kpts = results[0].keypoints.xy.cpu().numpy()
        boxes = results[0].boxes.xywh.cpu().numpy()
        main_idx = np.argmax(boxes[:, 2] * boxes[:, 3])
        target_x = (
            kpts[main_idx][0][0] if kpts[main_idx][0][0] > 0 else boxes[main_idx][0]
        )

    if target_x is not None:
        diff = target_x - smooth_x

        # ЛОГИКА АКТИВАЦИИ
        if not is_moving:
            # Если камера стоит, проверяем: не пора ли проснуться?
            if abs(diff) > trigger_threshold:
                is_moving = True

        # ЛОГИКА ДВИЖЕНИЯ
        if is_moving:
            # Если проснулись — плавно едем к цели
            smooth_x = (ease_speed * target_x) + (1 - ease_speed) * smooth_x

            # Если доехали достаточно близко — засыпаем
            if abs(target_x - smooth_x) < stop_threshold:
                is_moving = False

    # Фиксация и кроп
    min_center, max_center = target_w / 2, width - target_w / 2
    smooth_x = np.clip(smooth_x, min_center, max_center)

    # Применяем целое число только для отрисовки, чтобы не было "дрожания" в 1 пиксель
    x1 = int(round(smooth_x - target_w / 2))

    cropped_frame = frame[0:height, x1 : x1 + target_w]
    out.write(cropped_frame)

    cv2.imshow("Smart Tripod Slicer", cropped_frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
out.release()
cv2.destroyAllWindows()

# Сборка со звуком
subprocess.run(
    [
        "ffmpeg",
        "-i",
        temp_output_path,
        "-i",
        input_path,
        "-c:v",
        "libx264",
        "-crf",
        "18",
        "-preset",
        "slow",
        "-c:a",
        "aac",
        "-map",
        "0:v:0",
        "-map",
        "1:a:0?",
        "-shortest",
        "-y",
        output_path,
    ],
    check=True,
)
if os.path.exists(temp_output_path):
    os.remove(temp_output_path)
print(f"Готово! Видео в режиме штатива: {output_path}")
