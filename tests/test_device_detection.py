"""
Тестирование автоматической детекции устройства и fallback на CPU.
"""

from src.logger import logger
from src.transcription.device_utils import detect_device, get_device_info
from src.transcription.transcriber import VideoTranscriber


def test_device_detection():
    """Тестирует детекцию устройства."""

    logger.info("=" * 60)
    logger.info("Тестирование детекции устройства")
    logger.info("=" * 60)

    # Получаем информацию об устройстве
    info = get_device_info()
    logger.info("Информация об устройстве:")
    logger.info(f"  CUDA доступна: {info['cuda_available']}")
    logger.info(f"  Количество GPU: {info['cuda_device_count']}")

    if info['cuda_available']:
        logger.info(f"  GPU: {info['cuda_device_name']}")
        logger.info(f"  CUDA версия: {info['cuda_version']}")
    else:
        logger.warning("  CUDA недоступна")

    # Автоматическая детекция
    device, compute_type = detect_device()
    logger.info(f"\nАвтоматически выбрано: {device} с compute_type={compute_type}")

    logger.info("=" * 60)


def test_cuda_fallback():
    """Тестирует fallback с CUDA на CPU."""

    logger.info("=" * 60)
    logger.info("Тестирование CUDA fallback")
    logger.info("=" * 60)

    try:
        # Намеренно пытаемся загрузить на CUDA
        # Если cuDNN не работает, должен быть fallback на CPU
        logger.info("Попытка загрузки на CUDA...")
        transcriber = VideoTranscriber(
            model_size="tiny",
            device="cuda",
            compute_type="float16"
        )

        logger.success(f"Модель загружена на: {transcriber.device}")
        logger.info("=" * 60)

    except Exception as e:
        logger.exception(f"Критическая ошибка: {e}")


if __name__ == "__main__":
    test_device_detection()
    print()
    test_cuda_fallback()
