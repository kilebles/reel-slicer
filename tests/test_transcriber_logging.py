"""
Тест логирования в модуле transcriber без полной транскрипции.
"""

from src.logger import logger

def test_transcriber_import():
    """Проверяет импорт и инициализацию transcriber с логированием."""

    logger.info("=" * 60)
    logger.info("Тестирование импорта VideoTranscriber с логированием")
    logger.info("=" * 60)

    try:
        from src.transcription.transcriber import VideoTranscriber
        logger.success("VideoTranscriber успешно импортирован")

        # Тестируем инициализацию (это загрузит модель)
        logger.info("Попытка инициализации VideoTranscriber...")
        logger.warning("Это может занять время при первой загрузке модели")

        # Используем небольшую модель для быстрого теста
        # Для CPU нужно использовать int8 вместо float16
        transcriber = VideoTranscriber(
            model_size="tiny",
            device="cpu",
            compute_type="int8"
        )

        logger.success("VideoTranscriber успешно инициализирован!")
        logger.info("=" * 60)
        logger.success("Все компоненты работают с логированием!")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Ошибка при тестировании: {e}")
        raise

if __name__ == "__main__":
    test_transcriber_import()
