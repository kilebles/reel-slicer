import json
import re
import signal
import sys
from pathlib import Path
from typing import Dict, List

from src.logger import logger
from src.segmentation.analyzer import ViralSegmentAnalyzer
from src.transcription.device_utils import detect_device, get_device_info
from src.transcription.transcriber import VideoTranscriber
from src.transcription.short_transcriber import SubtitleGenerator
from src.video_processing import VideoCutter, VideoReframer, GifOverlay
from settings import GIF_OVERLAY, SUBTITLES, REFRAMING


shutdown_requested = False


def transliterate_to_slug(text: str, max_words: int = 3) -> str:
    """
    Преобразует русский текст в slug для имени файла

    Args:
        text: исходный текст
        max_words: максимальное количество слов для slug

    Returns:
        транслитерированный slug (например "moskovskie_sugrobi")
    """
    translit_map = {
        'а': 'a', 'б': 'b', 'в': 'v', 'г': 'g', 'д': 'd', 'е': 'e', 'ё': 'yo',
        'ж': 'zh', 'з': 'z', 'и': 'i', 'й': 'y', 'к': 'k', 'л': 'l', 'м': 'm',
        'н': 'n', 'о': 'o', 'п': 'p', 'р': 'r', 'с': 's', 'т': 't', 'у': 'u',
        'ф': 'f', 'х': 'h', 'ц': 'ts', 'ч': 'ch', 'ш': 'sh', 'щ': 'sch',
        'ъ': '', 'ы': 'y', 'ь': '', 'э': 'e', 'ю': 'yu', 'я': 'ya'
    }

    text = text.lower()
    text = re.sub(r'[«»"""„]', '', text)
    text = re.sub(r'[^\w\s]', ' ', text)

    words = [word for word in text.split() if len(word) > 2][:max_words]

    result = []
    for word in words:
        transliterated = ''.join(translit_map.get(char, char) for char in word)
        result.append(transliterated)

    return '_'.join(result)


def get_segment_mapping(analysis_path: Path, output_dir: Path) -> Dict[str, Dict]:
    """
    Создает маппинг между файлами видео и данными сегментов

    Returns:
        dict с ключами - путями к файлам и значениями - данными сегмента
    """
    if not analysis_path.exists():
        logger.warning(f"Файл анализа не найден: {analysis_path}")
        return {}

    with open(analysis_path, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)

    segments = analysis_data.get('segments', [])

    video_files = sorted(output_dir.glob("segment_*.mp4"))

    mapping = {}
    for idx, (video_file, segment) in enumerate(zip(video_files, segments), 1):
        slug = transliterate_to_slug(segment['hook'])
        mapping[str(video_file)] = {
            'segment_data': segment,
            'slug': slug,
            'index': idx
        }

    return mapping


def process_reframing(output_dir: Path, cropped_dir: Path, analysis_path: Path) -> List[Path]:
    """
    Обрабатывает все видео из output_dir с вертикальной обрезкой

    Args:
        output_dir: директория с нарезанными сегментами
        cropped_dir: директория для сохранения обрезанных видео
        analysis_path: путь к файлу анализа

    Returns:
        список путей к обработанным файлам
    """
    cropped_dir.mkdir(parents=True, exist_ok=True)

    mapping = get_segment_mapping(analysis_path, output_dir)

    if not mapping:
        logger.warning("Нет файлов для обработки в reframing")
        return []

    logger.info("\n" + "=" * 60)
    logger.info("Начало вертикальной обрезки видео (reframing)")
    logger.info("=" * 60)

    reframer = VideoReframer(
        trigger_threshold=REFRAMING["trigger_threshold"],
        stop_threshold=REFRAMING["stop_threshold"],
        ease_speed=REFRAMING["ease_speed"]
    )

    processed_files = []
    skipped_files = []

    for video_path_str, info in mapping.items():
        video_path = Path(video_path_str)
        slug = info['slug']

        new_input_name = f"segmented_{slug}.mp4"
        new_input_path = output_dir / new_input_name

        output_name = f"{slug}_cropped.mp4"
        output_path = cropped_dir / output_name

        if output_path.exists():
            logger.info(f"Пропускаем {slug} - уже обработан")
            skipped_files.append(output_path)
            continue

        if video_path != new_input_path and not new_input_path.exists():
            video_path.rename(new_input_path)
            logger.debug(f"Переименовано: {video_path.name} -> {new_input_path.name}")
            video_path = new_input_path
        elif new_input_path.exists():
            video_path = new_input_path

        try:
            logger.info(f"\nОбработка сегмента: {slug}")
            logger.info(f"  Hook: {info['segment_data']['hook'][:60]}...")
            logger.info(f"  Virality score: {info['segment_data']['virality_score']}/10")

            result_path = reframer.reframe_video(
                input_path=video_path,
                output_path=output_path
            )
            processed_files.append(result_path)

        except Exception as e:
            logger.error(f"Ошибка при обработке {video_path.name}: {e}")
            continue

    logger.info("\n" + "=" * 60)
    logger.info("Результаты reframing:")
    logger.info(f"  Обработано новых видео: {len(processed_files)}")
    logger.info(f"  Пропущено (уже существуют): {len(skipped_files)}")
    logger.info(f"  Папка вывода: {cropped_dir}")
    logger.info("=" * 60)

    if processed_files:
        logger.info("\nНовые обработанные файлы:")
        for file in processed_files[:5]:
            logger.info(f"  - {file.name}")
        if len(processed_files) > 5:
            logger.info(f"  ... и ещё {len(processed_files) - 5} файлов")

    return processed_files


def process_subtitles(
    cropped_dir: Path,
    subtitled_dir: Path,
    device: str = "cuda",
    compute_type: str = "float16",
) -> List[Path]:
    """
    Добавляет субтитры на обрезанные видео

    Args:
        cropped_dir: директория с обрезанными видео
        subtitled_dir: директория для видео с субтитрами
        device: устройство для Whisper
        compute_type: тип вычислений

    Returns:
        список путей к видео с субтитрами
    """
    subtitled_dir.mkdir(parents=True, exist_ok=True)

    cropped_videos = sorted(cropped_dir.glob("*_cropped.mp4"))

    if not cropped_videos:
        logger.warning("Нет обрезанных видео для добавления субтитров")
        return []

    logger.info("\n" + "=" * 60)
    logger.info("Начало добавления субтитров на видео")
    logger.info("=" * 60)

    subtitle_gen = SubtitleGenerator(
        device=device,
        compute_type=compute_type,
    )

    processed_files = []
    skipped_files = []

    for video_path in cropped_videos:
        base_name = video_path.stem.replace("_cropped", "")
        output_name = f"{base_name}_subtitled.mp4"
        output_path = subtitled_dir / output_name

        logger.debug(f"Проверка: {output_path} exists={output_path.exists()}")

        if output_path.exists():
            logger.info(f"Пропускаем {base_name} - субтитры уже добавлены")
            skipped_files.append(output_path)
            continue

        try:
            logger.info(f"\nДобавление субтитров: {base_name}")

            result_path = subtitle_gen.add_subtitles(
                video_path=video_path, output_path=output_path, language="ru"
            )
            processed_files.append(result_path)

        except Exception as e:
            logger.error(f"Ошибка при добавлении субтитров на {video_path.name}: {e}")
            continue

    logger.info("\n" + "=" * 60)
    logger.info("Результаты добавления субтитров:")
    logger.info(f"  Обработано новых видео: {len(processed_files)}")
    logger.info(f"  Пропущено (уже существуют): {len(skipped_files)}")
    logger.info(f"  Папка вывода: {subtitled_dir}")
    logger.info("=" * 60)

    if processed_files:
        logger.info("\nНовые видео с субтитрами:")
        for file in processed_files[:5]:
            logger.info(f"  - {file.name}")
        if len(processed_files) > 5:
            logger.info(f"  ... и ещё {len(processed_files) - 5} файлов")

    return processed_files


def process_gif_overlay(
    subtitled_dir: Path,
    final_dir: Path,
    gif_frames_dir: Path,
) -> List[Path]:
    """
    Накладывает анимированную гифку на видео с субтитрами

    Args:
        subtitled_dir: директория с видео с субтитрами
        final_dir: директория для финальных видео
        gif_frames_dir: директория с PNG фреймами для гифки
        start_time: время начала анимации в секундах

    Returns:
        список путей к финальным видео
    """
    final_dir.mkdir(parents=True, exist_ok=True)

    subtitled_videos = sorted(subtitled_dir.glob("*_subtitled.mp4"))

    if not subtitled_videos:
        logger.warning("Нет видео с субтитрами для наложения гифки")
        return []

    logger.info("\n" + "=" * 60)
    logger.info("Начало наложения анимированной гифки на видео")
    logger.info("=" * 60)

    gif_overlay = GifOverlay(
        frames_dir=gif_frames_dir,
        start_time=GIF_OVERLAY["start_time"],
        frame_duration=GIF_OVERLAY["frame_duration"],
        smooth_transitions=GIF_OVERLAY["smooth_transitions"],
        x_position=GIF_OVERLAY["x_position"],
        y_position=GIF_OVERLAY["y_position"],
        scale=GIF_OVERLAY["scale"],
    )

    processed_files = []
    skipped_files = []

    for video_path in subtitled_videos:
        base_name = video_path.stem.replace("_subtitled", "")
        output_name = f"{base_name}_final.mp4"
        output_path = final_dir / output_name

        if output_path.exists():
            logger.info(f"Пропускаем {base_name} - финальное видео уже существует")
            skipped_files.append(output_path)
            continue

        try:
            logger.info(f"\nНаложение гифки на: {base_name}")

            result_path = gif_overlay.overlay_on_video(
                input_path=video_path,
                output_path=output_path,
            )
            processed_files.append(result_path)

        except Exception as e:
            logger.error(f"Ошибка при наложении гифки на {video_path.name}: {e}")
            continue

    logger.info("\n" + "=" * 60)
    logger.info("Результаты наложения гифки:")
    logger.info(f"  Обработано новых видео: {len(processed_files)}")
    logger.info(f"  Пропущено (уже существуют): {len(skipped_files)}")
    logger.info(f"  Папка вывода: {final_dir}")
    logger.info("=" * 60)

    if processed_files:
        logger.info("\nФинальные видео:")
        for file in processed_files[:5]:
            logger.info(f"  - {file.name}")
        if len(processed_files) > 5:
            logger.info(f"  ... и ещё {len(processed_files) - 5} файлов")

    return processed_files


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
    logger.info("Информация об устройстве:")
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

        if analysis_path.exists():
            logger.info(f"Анализ уже существует: {analysis_path}")
            logger.info("Пропускаем этап анализа")

            analyzer = ViralSegmentAnalyzer()
            segments = analyzer.load_analysis(analysis_path)
        else:
            logger.info("\n" + "=" * 60)
            logger.info("Начало поиска вирусных фрагментов с помощью Claude AI")
            logger.info("=" * 60)

            analyzer = ViralSegmentAnalyzer()
            segments = analyzer.analyze(transcript_path)
            analyzer.save_analysis(segments, analysis_path)

            logger.info("\n" + "=" * 60)
            logger.info("Результаты анализа:")
            logger.info(f"  Найдено вирусных фрагментов: {len(segments)}")
            if segments:
                avg_score = sum(s.virality_score for s in segments) / len(segments)
                avg_duration = sum(s.duration for s in segments) / len(segments)
                logger.info(f"  Средний virality score: {avg_score:.1f}/10")
                logger.info(f"  Средняя длительность: {avg_duration:.1f}с")
            logger.info(f"  Сохранено: {analysis_path}")
            logger.info("=" * 60)

        if segments:
            logger.info("\nТоп вирусных фрагментов:")
            for i, seg in enumerate(segments[:5], 1):
                logger.info(
                    f"\n{i}. [{seg.start:.1f}s - {seg.end:.1f}s] ({seg.duration:.1f}с) "
                    f"Score: {seg.virality_score}/10"
                )
                logger.info(f"   Hook: {seg.hook}")
                logger.info(f"   Punchline: {seg.punchline}")
                logger.info(f"   Эмоция: {seg.emotion}")
                logger.info(f"   Причина: {seg.reason}")

            if len(segments) > 5:
                logger.info(f"\n... и ещё {len(segments) - 5} фрагментов")
        else:
            logger.warning("\nВирусных фрагментов не найдено!")

        if shutdown_requested:
            return

        if segments:
            logger.info("\n" + "=" * 60)
            logger.info("Начало нарезки видео на сегменты")
            logger.info("=" * 60)

            cutter = VideoCutter(
                video_path=str(video_path),
                analysis_path=str(analysis_path),
                output_dir="data/output"
            )
            cutter.load_analysis()
            output_files = cutter.cut_all_segments(
                name_pattern="segment_{index:02d}_{emotion}_score{virality_score}.mp4"
            )

            logger.info("\n" + "=" * 60)
            logger.info("Результаты нарезки:")
            logger.info(f"  Создано видео фрагментов: {len(output_files)}")
            logger.info(f"  Папка вывода: data/output")
            logger.info("=" * 60)

            if output_files:
                logger.info("\nСозданные файлы:")
                for file in output_files[:5]:
                    logger.info(f"  - {file.name}")
                if len(output_files) > 5:
                    logger.info(f"  ... и ещё {len(output_files) - 5} файлов")

        if shutdown_requested:
            return

        output_dir = Path("data/output")
        cropped_dir = Path("data/cropped")

        if output_dir.exists() and list(output_dir.glob("*.mp4")):
            cropped_files = process_reframing(
                output_dir=output_dir,
                cropped_dir=cropped_dir,
                analysis_path=analysis_path
            )

            if cropped_files:
                logger.info("\nВертикальная обрезка завершена успешно!")

        if shutdown_requested:
            return

        subtitled_dir = Path("data/subtitled")

        if cropped_dir.exists() and list(cropped_dir.glob("*_cropped.mp4")):
            subtitled_files = process_subtitles(
                cropped_dir=cropped_dir,
                subtitled_dir=subtitled_dir,
                device=device,
                compute_type=compute_type,
            )

            if subtitled_files:
                logger.info("\nДобавление субтитров завершено успешно!")

        if shutdown_requested:
            return

        final_dir = Path("data/final")
        gif_frames_dir = Path("data/gif_frames")

        if subtitled_dir.exists() and list(subtitled_dir.glob("*_subtitled.mp4")):
            if gif_frames_dir.exists() and list(gif_frames_dir.glob("*.png")):
                final_files = process_gif_overlay(
                    subtitled_dir=subtitled_dir,
                    final_dir=final_dir,
                    gif_frames_dir=gif_frames_dir,
                )

                if final_files:
                    logger.info("\nНаложение гифки завершено успешно!")
            else:
                logger.warning(f"Директория с фреймами гифки не найдена или пуста: {gif_frames_dir}")

        logger.success("\nОбработка завершена успешно!")

    except KeyboardInterrupt:
        logger.warning("\nПрограмма прервана пользователем")
        sys.exit(0)
    except Exception as e:
        logger.exception(f"Критическая ошибка при выполнении: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()