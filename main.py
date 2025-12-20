import signal
import sys
from pathlib import Path

from src.logger import logger
from src.segmentation.analyzer import TranscriptAnalyzer
from src.transcription.device_utils import detect_device, get_device_info
from src.transcription.transcriber import VideoTranscriber


shutdown_requested = False


def signal_handler(signum, frame):
    """Обрабатывает сигналы прерывания для корректного завершения программы."""
    global shutdown_requested
    signal_name = signal.Signals(signum).name
    logger.warning(f"\n{signal_name} получен. Корректное завершение программы...")
    shutdown_requested = True
    sys.exit(0)


def main():
    
    signal.signal(signal.SIGINT, signal_handler)  
    signal.signal(signal.SIGTERM, signal_handler)  

    logger.info("=" * 60)
    logger.info("Запуск Reel-Slicer")
    logger.info("=" * 60)

    
    device_info = get_device_info()
    logger.info(f"Информация об устройстве:")
    logger.info(f"  CUDA доступна: {device_info['cuda_available']}")
    if device_info['cuda_available']:
        logger.info(f"  Устройство: {device_info['cuda_device_name']}")
        logger.info(f"  CUDA версия: {device_info['cuda_version']}")

    
    device, compute_type = detect_device()
    logger.info(f"Выбрано устройство: {device} с compute_type={compute_type}")

    video_path = Path("data/video.mp4")
    transcript_path = Path("data/transcript.json")
    analysis_path = Path("data/analysis.json")

    try:
        
        if transcript_path.exists():
            logger.info(f"Транскрипция уже существует: {transcript_path}")
            logger.info("Пропускаем этап транскрипции")
        else:
            if not video_path.exists():
                logger.error(f"Видео не найдено: {video_path}")
                return

            logger.info(f"Обработка видео: {video_path.name}")

            transcriber = VideoTranscriber(
                model_size="medium",
                device="cuda",
                compute_type="float16"  
            )
            transcript = transcriber.transcribe(video_path, language="ru")

            transcriber.save_transcript(transcript, transcript_path)

            logger.info("=" * 60)
            logger.info("Результаты транскрипции:")
            logger.info(f"  Длительность: {transcript.duration:.1f}с")
            logger.info(f"  Сегментов: {len(transcript.segments)}")
            logger.info(f"  Слов: {sum(len(s.words) for s in transcript.segments)}")
            logger.info(f"  Сохранено: {transcript_path}")
            logger.info("=" * 60)

            logger.info("\nПервые сегменты:")
            for seg in transcript.segments[:3]:
                logger.info(f"[{seg.start:.1f}s - {seg.end:.1f}s] {seg.text}")

        if shutdown_requested:
            return

        
        logger.info("\n" + "=" * 60)
        logger.info("Начало анализа транскрипции с помощью Claude AI")
        logger.info("=" * 60)

        analyzer = TranscriptAnalyzer()
        analysis = analyzer.analyze(transcript_path)
        analyzer.save_analysis(analysis, analysis_path)

        logger.info("\n" + "=" * 60)
        logger.info("Результаты анализа:")
        logger.info(f"  Оригинальная длительность: {analysis['original_duration']:.1f}с")
        logger.info(f"  Оригинальных сегментов: {analysis['original_segments']}")
        logger.info(f"  Найдено логических сегментов: {analysis['identified_segments']}")
        logger.info(f"  Сохранено: {analysis_path}")
        logger.info("=" * 60)

        logger.info("\nНайденные сегменты:")
        for i, seg in enumerate(analysis['segments'][:5], 1):
            logger.info(
                f"{i}. [{seg['start']:.1f}s - {seg['end']:.1f}s] "
                f"{seg['topic']} - {seg['reason']}"
            )

        if len(analysis['segments']) > 5:
            logger.info(f"... и ещё {len(analysis['segments']) - 5} сегментов")

        logger.success("\nОбработка завершена успешно!")

    except KeyboardInterrupt:
        logger.warning("\nПрограмма прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Критическая ошибка при выполнении: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()