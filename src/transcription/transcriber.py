"""Модуль транскрибации видео с использованием faster-whisper."""

import json

from faster_whisper import WhisperModel
from pathlib import Path
from dataclasses import dataclass, asdict
from src.logger import logger


@dataclass
class Word:
    """
    Отдельное слово с временными метками.

    Attributes:
        word: Текст слова
        start: Начальное время в секундах
        end: Конечное время в секундах
        probability: Вероятность распознавания (0.0-1.0)
    """
    word: str
    start: float
    end: float
    probability: float


@dataclass
class Segment:
    """
    Сегмент транскрипции, содержащий несколько слов.

    Attributes:
        start: Начальное время сегмента в секундах
        end: Конечное время сегмента в секундах
        text: Полный текст сегмента
        words: Список объектов Word в сегменте
    """
    start: float
    end: float
    text: str
    words: list[Word]


@dataclass
class Transcript:
    """
    Полная транскрипция видео.

    Attributes:
        language: Код языка (например, 'ru', 'en')
        duration: Общая длительность видео в секундах
        segments: Список сегментов транскрипции
    """
    language: str
    duration: float
    segments: list[Segment]
    
    def to_dict(self) -> dict:
        """Преобразует Transcript в словарь для сериализации в JSON."""
        return {
            "language": self.language,
            "duration": self.duration,
            "segments": [
                {
                    **asdict(seg),
                    "words": [asdict(w) for w in seg.words]
                }
                for seg in self.segments
            ]
        }


class VideoTranscriber:
    """
    Транскрибатор видео с использованием модели Whisper.

    Поддерживает автоматический fallback с CUDA на CPU при ошибках.
    Создает word-level timestamps для точной сегментации.
    """
    def __init__(
        self,
        model_size: str = "medium",
        device: str = "cuda",
        compute_type: str = "float16"
    ):
        """
        Инициализирует транскрибатор с заданными параметрами.

        Args:
            model_size: Размер модели Whisper ("tiny", "base", "small", "medium", "large")
            device: Устройство для вычислений ("cuda" или "cpu")
            compute_type: Тип вычислений ("float16" для CUDA, "int8" для CPU)
        """
        logger.info(f"Инициализация VideoTranscriber: model={model_size}, device={device}, compute_type={compute_type}")

        # Попытка загрузки с указанным device
        try:
            self.model = WhisperModel(
                model_size,
                device=device,
                compute_type=compute_type
            )
            logger.success(f"Модель {model_size} успешно загружена на {device}")
            self.device = device
            self.compute_type = compute_type

        except Exception as e:
            logger.error(f"Ошибка загрузки модели на {device}: {e}")

            # Если была попытка использовать CUDA, пробуем CPU
            if device == "cuda":
                logger.warning("Попытка fallback на CPU с compute_type=int8...")
                try:
                    self.model = WhisperModel(
                        model_size,
                        device="cpu",
                        compute_type="int8"
                    )
                    logger.success(f"Модель {model_size} успешно загружена на CPU (fallback)")
                    self.device = "cpu"
                    self.compute_type = "int8"
                except Exception as cpu_error:
                    logger.error(f"Ошибка загрузки модели на CPU: {cpu_error}")
                    raise
            else:
                raise
    
    def transcribe(
        self,
        video_path: str | Path,
        language: str = "ru"
    ) -> Transcript:
        """
        Транскрибирует видео с word-level timestamps.

        Args:
            video_path: Путь к видеофайлу
            language: Код языка аудио (по умолчанию "ru")

        Returns:
            Объект Transcript с полной транскрипцией

        Raises:
            Exception: При ошибках чтения файла или транскрипции
        """
        logger.info(f"Начало транскрипции: {video_path}, язык={language}")

        try:
            segments_iter, info = self.model.transcribe(
                str(video_path),
                language=language,
                beam_size=5,
                word_timestamps=True,
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=500)
            )

            logger.info(f"Обнаружен язык: {info.language}, длительность: {info.duration:.1f}с")

            segments = []
            for idx, seg in enumerate(segments_iter, 1):
                words = [
                    Word(
                        word=w.word,
                        start=w.start,
                        end=w.end,
                        probability=w.probability
                    )
                    for w in seg.words
                ]

                segments.append(Segment(
                    start=seg.start,
                    end=seg.end,
                    text=seg.text.strip(),
                    words=words
                ))

                if idx % 10 == 0:
                    logger.debug(f"Обработано сегментов: {idx}")

            transcript = Transcript(
                language=info.language,
                duration=info.duration,
                segments=segments
            )

            logger.success(f"Транскрипция завершена: {len(segments)} сегментов, {sum(len(s.words) for s in segments)} слов")
            return transcript

        except Exception as e:
            logger.error(f"Ошибка при транскрипции {video_path}: {e}")
            raise
    
    def save_transcript(self, transcript: Transcript, output_path: str | Path):
        """
        Сохраняет транскрипт в JSON-файл.

        Args:
            transcript: Объект Transcript для сохранения
            output_path: Путь к выходному JSON-файлу

        Raises:
            Exception: При ошибках записи файла
        """
        try:
            logger.info(f"Сохранение транскрипта в {output_path}")
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(transcript.to_dict(), f, ensure_ascii=False, indent=2)
            logger.success(f"Транскрипт сохранен: {output_path}")
        except Exception as e:
            logger.error(f"Ошибка сохранения транскрипта в {output_path}: {e}")
            raise
    
    @staticmethod
    def load_transcript(path: str | Path) -> Transcript:
        """
        Загружает транскрипт из JSON-файла.

        Args:
            path: Путь к JSON-файлу с транскрипцией

        Returns:
            Восстановленный объект Transcript

        Raises:
            Exception: При ошибках чтения или парсинга файла
        """
        try:
            logger.info(f"Загрузка транскрипта из {path}")
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)

            segments = [
                Segment(
                    start=seg["start"],
                    end=seg["end"],
                    text=seg["text"],
                    words=[Word(**w) for w in seg["words"]]
                )
                for seg in data["segments"]
            ]

            transcript = Transcript(
                language=data["language"],
                duration=data["duration"],
                segments=segments
            )

            logger.success(f"Транскрипт загружен: {len(segments)} сегментов")
            return transcript

        except Exception as e:
            logger.error(f"Ошибка загрузки транскрипта из {path}: {e}")
            raise