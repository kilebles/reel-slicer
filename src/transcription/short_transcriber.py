"""Модуль для создания элегантных, стабильных и анимированных субтитров."""

import os
from pathlib import Path
from typing import List, Tuple, Optional

# Настройка для ImageMagick (обязательно для Windows, если используется этот движок)
if os.name == "nt":
    os.environ["IMAGEMAGICK_BINARY"] = (
        r"C:\Program Files\ImageMagick-7.1.1-Q16-HDRI\magick.exe"
    )

from moviepy import VideoFileClip, TextClip, CompositeVideoClip
import moviepy.video.fx as vfx

from src.logger import logger
from src.transcription.transcriber import VideoTranscriber, Transcript


class SubtitleGenerator:
    """
    Класс для генерации субтитров.
    Оптимизирован под MoviePy 2.1.1.
    Решает проблемы обрезки текста и прыгающего позиционирования.
    """

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cuda",
        compute_type: str = "float16",
        words_per_line: int = 2,  # Короткие фразы лучше читаются
        font_size: int = 32,  # Элегантный размер
        font: str = "impact.ttf",  # Жирный шрифт
        color: str = "white",  # Чистый белый цвет
        stroke_color: str = "black",
        stroke_width: int = 3,  # Тонкая аккуратная обводка
        position: Tuple[str, str] = ("center", "bottom"),
    ):
        self.transcriber = VideoTranscriber(
            model_size=model_size, device=device, compute_type=compute_type
        )
        self.words_per_line = words_per_line
        self.font_size = font_size
        self.color = color
        self.stroke_color = stroke_color
        self.stroke_width = stroke_width
        self.position = position
        self.font_path = self._get_font_path(font)

        logger.info(f"SubtitleGenerator готов. Шрифт: {self.font_path}")

    def _get_font_path(self, font_name: str) -> str:
        """Ищет шрифт в системе Windows."""
        fonts_dir = Path(r"C:\Windows\Fonts")
        paths = [
            fonts_dir / font_name,
            fonts_dir
            / "impact.ttf",  # Если Arial не нравится, Impact — стандарт Shorts
            fonts_dir / "impact.ttf",
        ]
        for p in paths:
            if p.exists():
                return str(p)
        return font_name

    def add_subtitles(
        self,
        video_path: Path | str,
        output_path: Path | str,
        language: str = "ru",
        transcript: Optional[Transcript] = None,
    ) -> Path:
        video_path, output_path = Path(video_path), Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        if transcript is None:
            logger.info(f"Транскрибация: {video_path.name}")
            transcript = self.transcriber.transcribe(video_path, language=language)

        logger.info("Создание субтитров...")
        subtitle_clips = self._create_subtitle_clips(transcript, video_path)

        if not subtitle_clips:
            logger.error("Не удалось создать ни одного клипа!")
            return video_path

        logger.info(f"Рендеринг видео с {len(subtitle_clips)} фразами...")
        self._render_video_with_subtitles(video_path, output_path, subtitle_clips)

        return output_path

    def _create_subtitle_clips(
        self, transcript: Transcript, video_path: Path
    ) -> List[TextClip]:
        all_words = []
        for segment in transcript.segments:
            all_words.extend(segment.words)

        subtitle_clips = []
        video = VideoFileClip(str(video_path))
        v_width, v_height = video.size
        video.close()

        # Параметры макета
        max_width = int(v_width * 0.8)
        # Большая фиксированная высота контейнера исключает обрезку обводки
        fixed_height = int(self.font_size * 4)
        # Точка на экране (Y), вокруг которой будет центрироваться текст
        center_y_pos = v_height - 200

        for idx, word in enumerate(all_words):
            text = word.word.strip().upper()
            if not text:
                continue

            start_t = word.start
            if idx + 1 < len(all_words):
                end_t = all_words[idx + 1].start
            else:
                end_t = word.end

            dur = max(0.1, end_t - start_t)

            try:
                txt_clip = TextClip(
                    text=text,
                    font=self.font_path,
                    font_size=self.font_size,
                    color=self.color,
                    stroke_color=self.stroke_color,
                    stroke_width=self.stroke_width,
                    method="caption",
                    size=(max_width, fixed_height),
                    text_align="center",
                )

                y_pos = center_y_pos - (fixed_height // 2)

                # Элегантный микровзрыв (зум)
                def zoom_effect(t):
                    # Длительность анимации: 0.15 сек
                    # Пик зума: 1.1 (всего на 10% больше)
                    if t < 0.15:
                        return 1.1 - (t / 0.15) * 0.1
                    return 1.0

                txt_clip = (
                    txt_clip.with_start(start_t)
                    .with_duration(dur)
                    .with_position(("center", y_pos))
                    # Resize в MoviePy 2.x лучше работает через lambda для плавности
                    .with_effects([vfx.Resize(lambda t: zoom_effect(t))])
                )

                subtitle_clips.append(txt_clip)

            except Exception as e:
                logger.warning(f"Ошибка создания клипа '{text}': {e}")
                continue

        return subtitle_clips

    def _render_video_with_subtitles(
        self, video_path: Path, output_path: Path, subtitle_clips: List[TextClip]
    ) -> None:
        video = VideoFileClip(str(video_path))
        final_video = CompositeVideoClip([video, *subtitle_clips])

        final_video.write_videofile(
            str(output_path),
            codec="libx264",
            audio_codec="aac",
            fps=video.fps,
            preset="medium",
            threads=os.cpu_count() or 4,
            logger=None,
        )

        video.close()
        final_video.close()
        for c in subtitle_clips:
            c.close()


def add_subtitles_to_video(video_path, output_path, **kwargs):
    """Обертка для внешнего вызова (необходима для работы всей системы)."""
    generator = SubtitleGenerator(**kwargs)
    return generator.add_subtitles(video_path, output_path)
