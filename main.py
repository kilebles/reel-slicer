from pathlib import Path
from src.transcription.transcriber import VideoTranscriber
from src.transcription.device_utils import detect_device, get_device_info
from src.logger import logger


def main():
    logger.info("=" * 60)
    logger.info("Запуск Reel-Slicer")
    logger.info("=" * 60)

    # Проверяем доступные устройства
    device_info = get_device_info()
    logger.info(f"Информация об устройстве:")
    logger.info(f"  CUDA доступна: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        logger.info(f"  Устройство: {device_info['cuda_device_name']}")
        logger.info(f"  CUDA версия: {device_info['cuda_version']}")

    # Автоматически выбираем лучшее устройство
    device, compute_type = detect_device()
    logger.info(f"Выбрано устройство: {device} с compute_type={compute_type}")

    video_path = Path("data/video.mp4")

    if not video_path.exists():
        logger.error(f"Видео не найдено: {video_path}")
        return

    logger.info(f"Обработка видео: {video_path.name}")

    try:
        transcriber = VideoTranscriber(
            model_size="medium",
            device="cuda",
            compute_type="float16" # Или "int8_float16" для экономии памяти
        )
        transcript = transcriber.transcribe(video_path, language="ru")

        output_path = Path("data/transcript.json")
        transcriber.save_transcript(transcript, output_path)

        logger.info("=" * 60)
        logger.info("Результаты транскрипции:")
        logger.info(f"  Длительность: {transcript.duration:.1f}с")
        logger.info(f"  Сегментов: {len(transcript.segments)}")
        logger.info(f"  Слов: {sum(len(s.words) for s in transcript.segments)}")
        logger.info(f"  Сохранено: {output_path}")
        logger.info("=" * 60)

        logger.info("\nПервые сегменты:")
        for seg in transcript.segments[:3]:
            logger.info(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")

        logger.success("Обработка завершена успешно!")

    except Exception as e:
        logger.exception(f"Критическая ошибка при выполнении: {e}")
        raise


if __name__ == "__main__":
    main()