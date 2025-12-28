"""
Настройка логирования для reel-slicer.
Использует loguru для удобного и понятного логирования.
"""

from loguru import logger
from pathlib import Path
import sys


def setup_logger(
    log_dir: str | Path = "logs",
    log_level: str = "INFO",
    rotation: str = "10 MB",
    retention: str = "7 days"
):
    """
    Настраивает логирование для приложения.

    Args:
        log_dir: Директория для сохранения логов
        log_level: Уровень логирования (DEBUG, INFO, WARNING, ERROR)
        rotation: Когда ротировать логи (по размеру или времени)
        retention: Как долго хранить старые логи
    """
    # Удаляем стандартный handler
    logger.remove()

    # Добавляем вывод в консоль с цветным форматированием
    logger.add(
        sys.stderr,
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level: <8}</level> | <cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        level=log_level,
        colorize=True
    )

    # Создаем директорию для логов если её нет
    log_path = Path(log_dir)
    log_path.mkdir(exist_ok=True)

    # Добавляем запись в файл (все логи)
    logger.add(
        log_path / "reel-slicer.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}",
        level="DEBUG",  # В файл пишем всё, даже DEBUG
        rotation=rotation,
        retention=retention,
        encoding="utf-8"
    )

    # Добавляем отдельный файл только для ошибок
    logger.add(
        log_path / "errors.log",
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {name}:{function}:{line} - {message}\n{exception}",
        level="ERROR",
        rotation=rotation,
        retention=retention,
        encoding="utf-8",
        backtrace=True,
        diagnose=True
    )

    logger.info(f"Логирование настроено. Логи сохраняются в: {log_path.absolute()}")
    return logger


# Создаем глобальный экземпляр логгера с дефолтными настройками
logger = setup_logger()
