import subprocess
from pathlib import Path
from typing import Optional
import tempfile

from src.logger import logger


class GifOverlay:
    """
    Класс для наложения анимированной гифки из PNG фреймов на видео
    """

    def __init__(
        self,
        frames_dir: Path | str,
        start_time: float = 5.0,
        frame_duration: float = 1.0,
        smooth_transitions: bool = True,
        x_position: int | str = 0,
        y_position: int = 0,
        scale: float = 1.0,
        letterbox_enabled: bool = False,
        letterbox_top: int = 0,
        letterbox_bottom: int = 0,
    ):
        """
        Args:
            frames_dir: директория с PNG фреймами для анимации
            start_time: время начала анимации в секундах
            frame_duration: длительность показа каждого кадра в секундах
            smooth_transitions: использовать плавные переходы между кадрами
            x_position: позиция X ("center", "left", "right" или число в пикселях)
            y_position: позиция Y ("top", "center", "bottom" или число в пикселях от верха)
            scale: масштаб гифки относительно ширины видео (0.3 = 30% ширины)
            letterbox_enabled: добавить черные полосы сверху/снизу
            letterbox_top: высота черной полосы сверху в пикселях
            letterbox_bottom: высота черной полосы снизу в пикселях
        """
        self.frames_dir = Path(frames_dir)
        self.start_time = start_time
        self.frame_duration = frame_duration
        self.smooth_transitions = smooth_transitions
        self.x_position = x_position
        self.y_position = y_position
        self.scale = scale
        self.letterbox_enabled = letterbox_enabled
        self.letterbox_top = letterbox_top
        self.letterbox_bottom = letterbox_bottom

        if not self.frames_dir.exists():
            raise FileNotFoundError(f"Директория с фреймами не найдена: {frames_dir}")

        self.frame_files = sorted(self.frames_dir.glob("*.png"))
        if not self.frame_files:
            raise ValueError(f"Не найдено PNG фреймов в {frames_dir}")

        logger.info("GifOverlay инициализирован")
        logger.info(f"  Директория фреймов: {self.frames_dir}")
        logger.info(f"  Количество фреймов: {len(self.frame_files)}")
        logger.info(f"  Начало анимации: {start_time}с")
        logger.info(f"  Длительность кадра: {frame_duration}с")
        logger.info(f"  Плавные переходы: {'Да' if smooth_transitions else 'Нет'}")
        logger.info(f"  Масштаб: {scale * 100}%")
        logger.info(f"  Позиция: X={x_position}, Y={y_position}")
        if letterbox_enabled:
            logger.info(f"  Letterbox: Верх={letterbox_top}px, Низ={letterbox_bottom}px")

    def overlay_on_video(
        self,
        input_path: Path | str,
        output_path: Path | str,
    ) -> Path:
        """
        Накладывает анимированную гифку на видео

        Args:
            input_path: путь к исходному видео
            output_path: путь для сохранения результата

        Returns:
            Path к видео с наложенной анимацией
        """
        input_path = Path(input_path)
        output_path = Path(output_path)

        if not input_path.exists():
            raise FileNotFoundError(f"Входное видео не найдено: {input_path}")

        output_path.parent.mkdir(parents=True, exist_ok=True)

        logger.info(f"Начало наложения анимации на: {input_path.name}")
        logger.info(f"  Выходной файл: {output_path}")

        try:
            self._overlay_with_ffmpeg(str(input_path), str(output_path))

            logger.success(f"Наложение анимации завершено: {output_path.name}")
            return output_path

        except Exception as e:
            logger.error(f"Ошибка при наложении анимации на {input_path.name}: {e}")
            raise

    def _overlay_with_ffmpeg(self, input_path: str, output_path: str) -> None:
        """
        Использует ffmpeg для создания анимации из PNG и наложения на видео
        """
        logger.info("  Наложение анимации с помощью FFmpeg...")

        frame_pattern = str(self.frames_dir / "%dframe.png")
        num_frames = len(self.frame_files)

        # Вычисляем FPS из frame_duration
        fps = 1.0 / self.frame_duration

        # Вычисляем позицию X
        if isinstance(self.x_position, str):
            if self.x_position == "center":
                x_expr = "(main_w-overlay_w)/2"
            elif self.x_position == "left":
                x_expr = "0"
            elif self.x_position == "right":
                x_expr = "main_w-overlay_w"
            else:
                x_expr = str(self.x_position)
        else:
            x_expr = str(self.x_position)

        # Вычисляем позицию Y
        if isinstance(self.y_position, str):
            if self.y_position == "center":
                y_expr = "(main_h-overlay_h)/2"
            elif self.y_position == "top":
                y_expr = "0"
            elif self.y_position == "bottom":
                y_expr = "main_h-overlay_h"
            else:
                y_expr = str(self.y_position)
        else:
            y_expr = str(self.y_position)

        # Создаем filter с масштабированием гифки
        scale_filter = f"scale=iw*{self.scale}:ih*{self.scale}"

        # Строим filter_complex
        if self.letterbox_enabled:
            # Добавляем letterbox к основному видео
            pad_height = f"ih+{self.letterbox_top}+{self.letterbox_bottom}"
            letterbox_filter = f"pad=iw:{pad_height}:0:{self.letterbox_top}:black"

            if self.smooth_transitions:
                smooth_filter = f"minterpolate=fps=30:mi_mode=blend"
                filter_complex = (
                    f"[0:v]{letterbox_filter}[padded];"
                    f"[1:v]{scale_filter},{smooth_filter}[scaled];"
                    f"[padded][scaled]overlay={x_expr}:{y_expr}:"
                    f"enable='gte(t,{self.start_time})':shortest=1[out]"
                )
            else:
                filter_complex = (
                    f"[0:v]{letterbox_filter}[padded];"
                    f"[1:v]{scale_filter}[scaled];"
                    f"[padded][scaled]overlay={x_expr}:{y_expr}:"
                    f"enable='gte(t,{self.start_time})':shortest=1[out]"
                )
        else:
            # Без letterbox (оригинальное поведение)
            if self.smooth_transitions:
                smooth_filter = f"minterpolate=fps=30:mi_mode=blend"
                filter_complex = (
                    f"[1:v]{scale_filter},{smooth_filter}[scaled];"
                    f"[0:v][scaled]overlay={x_expr}:{y_expr}:"
                    f"enable='gte(t,{self.start_time})':shortest=1[out]"
                )
            else:
                filter_complex = (
                    f"[1:v]{scale_filter}[scaled];"
                    f"[0:v][scaled]overlay={x_expr}:{y_expr}:"
                    f"enable='gte(t,{self.start_time})':shortest=1[out]"
                )

        ffmpeg_cmd = [
            "ffmpeg",
            "-i", input_path,
            "-stream_loop", "-1",
            "-framerate", str(fps),
            "-i", frame_pattern,
            "-filter_complex", filter_complex,
            "-map", "[out]",
            "-map", "0:a?",
            "-c:v", "libx264",
            "-crf", "23",
            "-preset", "ultrafast",
            "-c:a", "copy",
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

    def overlay_on_multiple(
        self,
        input_dir: Path | str,
        output_dir: Path | str,
        pattern: str = "*.mp4",
    ) -> list[Path]:
        """
        Накладывает анимацию на все видео в директории

        Args:
            input_dir: директория с исходными видео
            output_dir: директория для сохранения результатов
            pattern: шаблон для поиска видео файлов

        Returns:
            список путей к обработанным видео
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        video_files = sorted(input_dir.glob(pattern))

        if not video_files:
            logger.warning(f"Не найдено видео по шаблону {pattern} в {input_dir}")
            return []

        logger.info(f"Найдено видео для обработки: {len(video_files)}")

        processed_files = []
        for video_path in video_files:
            base_name = video_path.stem
            output_name = f"{base_name}_with_gif.mp4"
            output_path = output_dir / output_name

            if output_path.exists():
                logger.info(f"Пропускаем {base_name} - уже обработан")
                continue

            try:
                result_path = self.overlay_on_video(video_path, output_path)
                processed_files.append(result_path)
            except Exception as e:
                logger.error(f"Ошибка при обработке {video_path.name}: {e}")
                continue

        return processed_files
