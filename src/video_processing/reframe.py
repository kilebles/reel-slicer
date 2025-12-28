import os
import subprocess
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import torch
from ultralytics import YOLO

from src.logger import logger


class VideoReframer:
    """
    Класс для вертикальной обрезки видео с отслеживанием лица
    используя YOLO pose detection и алгоритм "умного штатива"
    """

    def __init__(
        self,
        trigger_threshold: int = 40,
        stop_threshold: int = 5,
        ease_speed: float = 0.12,
        target_aspect_ratio: tuple = (9, 16),
        conf_threshold: float = 0.5,
    ):
        """
        Args:
            trigger_threshold: смещение в пикселях для активации движения камеры
            stop_threshold: точность центрирования для остановки камеры
            ease_speed: скорость перестановки камеры (0-1, больше = быстрее)
            target_aspect_ratio: целевое соотношение сторон (ширина, высота)
            conf_threshold: порог уверенности для детекции YOLO
        """
        self.trigger_threshold = trigger_threshold
        self.stop_threshold = stop_threshold
        self.ease_speed = ease_speed
        self.target_aspect_ratio = target_aspect_ratio
        self.conf_threshold = conf_threshold

        self.device = "0" if torch.cuda.is_available() else "cpu"
        self.model = YOLO("yolov8m-pose.pt")

        logger.info("VideoReframer инициализирован")
        logger.info(f"  Устройство: {self.device}")
        logger.info(f"  Trigger threshold: {trigger_threshold}px")
        logger.info(f"  Ease speed: {ease_speed}")

    def reframe_video(
        self,
        input_path: Path | str,
        output_path: Path | str,
        temp_dir: Optional[Path | str] = None,
    ) -> Path:
        """
        Обрабатывает видео с вертикальной обрезкой и добавлением звука

        Args:
            input_path: путь к исходному видео
            output_path: путь для сохранения результата
            temp_dir: директория для временных файлов (по умолчанию рядом с output_path)

        Returns:
            Path к обработанному видео
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Входное видео не найдено: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        if temp_dir is None:
            temp_dir = output_path.parent
        else:
            temp_dir = Path(temp_dir)
            temp_dir.mkdir(parents=True, exist_ok=True)

        temp_output_path = temp_dir / f"temp_{output_path.stem}.mp4"

        logger.info(f"Начало обработки: {input_path.name}")
        logger.info(f"  Выходной файл: {output_path}")

        try:
            self._process_video(str(input_path), str(temp_output_path))
            self._add_audio_and_encode(
                str(temp_output_path), str(input_path), str(output_path)
            )

            if temp_output_path.exists():
                temp_output_path.unlink()
                logger.debug(f"Временный файл удален: {temp_output_path}")

            logger.success(f"Обработка завершена: {output_path.name}")
            return output_path

        except Exception as e:
            logger.error(f"Ошибка при обработке {input_path.name}: {e}")
            if temp_output_path.exists():
                logger.info(f"Временный файл сохранен: {temp_output_path}")
            raise

    def _process_video(self, input_path: str, output_path: str) -> None:
        """Обрабатывает видео с применением алгоритма умного штатива"""
        cap = cv2.VideoCapture(input_path)

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        target_w = int(height * self.target_aspect_ratio[0] / self.target_aspect_ratio[1])
        if target_w % 2 != 0:
            target_w -= 1

        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(output_path, fourcc, fps, (target_w, height))

        smooth_x = width // 2
        is_moving = False

        logger.info(f"  Параметры видео: {width}x{height} @ {fps}fps")
        logger.info(f"  Целевая ширина: {target_w}px (соотношение {self.target_aspect_ratio[0]}:{self.target_aspect_ratio[1]})")
        logger.info(f"  Всего кадров: {total_frames}")

        frame_count = 0
        log_interval = max(1, total_frames // 20)

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1

                results = self.model(
                    frame, verbose=False, conf=self.conf_threshold, device=self.device
                )
                target_x = None

                if len(results[0].keypoints) > 0:
                    kpts = results[0].keypoints.xy.cpu().numpy()
                    boxes = results[0].boxes.xywh.cpu().numpy()
                    main_idx = np.argmax(boxes[:, 2] * boxes[:, 3])

                    target_x = (
                        kpts[main_idx][0][0]
                        if kpts[main_idx][0][0] > 0
                        else boxes[main_idx][0]
                    )

                if target_x is not None:
                    diff = target_x - smooth_x

                    if not is_moving:
                        if abs(diff) > self.trigger_threshold:
                            is_moving = True

                    if is_moving:
                        smooth_x = (
                            self.ease_speed * target_x + (1 - self.ease_speed) * smooth_x
                        )

                        if abs(target_x - smooth_x) < self.stop_threshold:
                            is_moving = False

                min_center = target_w / 2
                max_center = width - target_w / 2
                smooth_x = np.clip(smooth_x, min_center, max_center)

                x1 = int(round(smooth_x - target_w / 2))

                cropped_frame = frame[0:height, x1 : x1 + target_w]
                out.write(cropped_frame)

                if frame_count % log_interval == 0:
                    progress = (frame_count / total_frames) * 100
                    logger.debug(f"  Прогресс: {progress:.1f}% ({frame_count}/{total_frames})")

        finally:
            cap.release()
            out.release()

        logger.info(f"  Обработано кадров: {frame_count}")

    def _add_audio_and_encode(
        self, video_path: str, audio_source_path: str, output_path: str
    ) -> None:
        """Добавляет аудио дорожку и перекодирует в H.264"""
        logger.info("  Добавление звука и перекодирование в H.264...")

        ffmpeg_cmd = [
            "ffmpeg",
            "-i",
            video_path,
            "-i",
            audio_source_path,
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
        ]

        try:
            result = subprocess.run(
                ffmpeg_cmd, check=True, capture_output=True, text=True
            )
            logger.debug("  FFmpeg завершен успешно")
        except subprocess.CalledProcessError as e:
            logger.error(f"Ошибка FFmpeg: {e.stderr}")
            raise
        except FileNotFoundError:
            logger.error("FFmpeg не найден. Установите ffmpeg и добавьте в PATH")
            raise
