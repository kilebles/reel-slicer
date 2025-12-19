"""
Утилиты для определения доступного устройства для транскрипции.
"""

import torch
from src.logger import logger


def detect_device() -> tuple[str, str]:
    """
    Автоматически определяет лучшее доступное устройство.

    Returns:
        tuple[str, str]: (device, compute_type)
            - ("cuda", "float16") если CUDA доступна
            - ("cpu", "int8") если только CPU
    """
    try:
        if torch.cuda.is_available():
            logger.info(f"Обнаружена CUDA: {torch.cuda.get_device_name(0)}")
            logger.info(f"CUDA версия: {torch.version.cuda}")
            return ("cuda", "float16")
        else:
            logger.warning("CUDA недоступна, используется CPU")
            return ("cpu", "int8")
    except Exception as e:
        logger.warning(f"Ошибка при проверке CUDA: {e}. Используется CPU")
        return ("cpu", "int8")


def get_device_info() -> dict:
    """
    Получает подробную информацию об устройстве.

    Returns:
        dict: Информация о доступных устройствах
    """
    info = {
        "cuda_available": False,
        "cuda_device_count": 0,
        "cuda_device_name": None,
        "cuda_version": None,
    }

    try:
        info["cuda_available"] = torch.cuda.is_available()

        if info["cuda_available"]:
            info["cuda_device_count"] = torch.cuda.device_count()
            info["cuda_device_name"] = torch.cuda.get_device_name(0)
            info["cuda_version"] = torch.version.cuda

    except Exception as e:
        logger.debug(f"Ошибка получения информации о CUDA: {e}")

    return info
