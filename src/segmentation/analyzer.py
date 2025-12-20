"""Анализ транскрипций для поиска вирусных фрагментов с помощью Claude AI."""

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path

from anthropic import Anthropic, RateLimitError

from src.logger import logger
from src.segmentation.prompts import (
    create_viral_segments_prompt,
    format_transcript_for_analysis,
)
from src.settings import settings
from src.transcription.transcriber import VideoTranscriber


@dataclass
class ViralSegment:
    """Представляет вирусный фрагмент видео."""

    start: float
    end: float
    hook: str
    punchline: str
    virality_score: int
    reason: str
    emotion: str

    @property
    def duration(self) -> float:
        """Длительность сегмента в секундах."""
        return self.end - self.start

    def is_valid(self) -> bool:
        """Проверяет валидность сегмента по критериям."""
        return (
            25 <= self.duration <= 60
            and self.virality_score >= 6
            and self.start < self.end
        )


class ViralSegmentAnalyzer:
    """Анализирует транскрипции для поиска вирусных фрагментов."""

    def __init__(self):
        self.client = Anthropic(api_key=settings.API_KEY)
        self.model = settings.ANTHROPIC_MODEL
        self.max_tokens = settings.MAX_TOKENS

    def _call_claude_api(self, prompt: str, max_retries: int = 3) -> dict:
        """Выполняет API-вызов к Claude с retry логикой."""
        retry_delay = 60

        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[{"role": "user", "content": prompt}],
                )

                response_text = message.content[0].text
                logger.debug(f"Claude response: {response_text[:200]}...")

                # Очистка от markdown
                response_text = response_text.strip()
                if response_text.startswith("```json"):
                    response_text = response_text[7:]
                elif response_text.startswith("```"):
                    response_text = response_text[3:]
                if response_text.endswith("```"):
                    response_text = response_text[:-3]
                response_text = response_text.strip()

                result = json.loads(response_text)
                return result

            except RateLimitError as e:
                if attempt < max_retries - 1:
                    logger.warning(
                        f"Rate limit reached. Waiting {retry_delay}s before retry "
                        f"(attempt {attempt + 1}/{max_retries})..."
                    )
                    time.sleep(retry_delay)
                    retry_delay *= 2
                else:
                    logger.error(f"Rate limit exceeded after {max_retries} attempts")
                    raise

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse Claude response as JSON: {e}")
                logger.error(f"Response was: {response_text}")
                raise
            except Exception as e:
                logger.error(f"Error calling Claude API: {e}")
                raise

    def _parse_segments(self, response: dict) -> list[ViralSegment]:
        """Парсит ответ Claude в список ViralSegment."""
        segments = []

        if "segments" not in response:
            logger.warning("No 'segments' key in Claude response")
            return segments

        for seg_data in response["segments"]:
            try:
                segment = ViralSegment(
                    start=float(seg_data["start"]),
                    end=float(seg_data["end"]),
                    hook=seg_data["hook"],
                    punchline=seg_data["punchline"],
                    virality_score=int(seg_data["virality_score"]),
                    reason=seg_data["reason"],
                    emotion=seg_data["emotion"],
                )

                if segment.is_valid():
                    segments.append(segment)
                    logger.debug(
                        f"Valid segment: {segment.duration:.1f}s, "
                        f"score={segment.virality_score}, hook='{segment.hook[:50]}...'"
                    )
                else:
                    logger.warning(
                        f"Invalid segment filtered out: {segment.duration:.1f}s, "
                        f"score={segment.virality_score}"
                    )

            except (KeyError, ValueError, TypeError) as e:
                logger.error(f"Failed to parse segment: {e}")
                logger.error(f"Segment data: {seg_data}")
                continue

        return segments

    def _align_to_word_boundaries(
        self, segments: list[ViralSegment], transcript_segments: list[dict]
    ) -> list[ViralSegment]:
        """Выравнивает временные метки сегментов по границам слов."""
        all_words = []
        for seg in transcript_segments:
            all_words.extend(seg.get("words", []))

        if not all_words:
            logger.warning("No words found in transcript, skipping alignment")
            return segments

        aligned_segments = []
        for seg in segments:
            start_word = self._find_hook_start(seg.hook, seg.start, all_words)
            end_word = self._find_nearest_word_end(seg.end, all_words)

            if start_word and end_word:
                new_start = start_word["start"]
                new_end = end_word["end"]

                if new_start < new_end:
                    aligned_seg = ViralSegment(
                        start=new_start,
                        end=new_end,
                        hook=seg.hook,
                        punchline=seg.punchline,
                        virality_score=seg.virality_score,
                        reason=seg.reason,
                        emotion=seg.emotion,
                    )
                    aligned_segments.append(aligned_seg)
                    logger.debug(
                        f"Aligned segment: {seg.start:.1f}->{new_start:.1f}, "
                        f"{seg.end:.1f}->{new_end:.1f}"
                    )
                else:
                    logger.warning(
                        f"Invalid alignment for segment {seg.start}-{seg.end}, skipping"
                    )
            else:
                logger.warning(
                    f"Could not align segment {seg.start}-{seg.end}, keeping original"
                )
                aligned_segments.append(seg)

        return aligned_segments

    def _find_hook_start(
        self, hook: str, target_time: float, words: list[dict]
    ) -> dict | None:
        """Находит начало хука в транскрипте."""
        if not words or not hook:
            return None

        hook_words = hook.split()[:3]

        nearest_word_idx = min(
            range(len(words)), key=lambda i: abs(words[i]["start"] - target_time)
        )

        search_start = max(0, nearest_word_idx - 20)
        search_end = min(len(words), nearest_word_idx + 5)

        for i in range(search_start, search_end):
            match_count = 0
            for j, hook_word in enumerate(hook_words):
                if i + j >= len(words):
                    break

                transcript_word = words[i + j]["word"].strip().lower()
                hook_word_clean = hook_word.strip('.,!?«»"').lower()

                if hook_word_clean in transcript_word or transcript_word in hook_word_clean:
                    match_count += 1
                else:
                    break

            if match_count >= 2:
                logger.debug(f"Found hook start at word {i}: '{words[i]['word']}'")
                return words[i]

        logger.warning(f"Could not find hook in transcript, using fallback")
        return self._find_nearest_sentence_start(target_time, words)

    def _find_nearest_sentence_start(
        self, target_time: float, words: list[dict]
    ) -> dict | None:
        """Находит начало предложения ближайшее к целевому времени."""
        if not words:
            return None

        nearest_word_idx = min(
            range(len(words)), key=lambda i: abs(words[i]["start"] - target_time)
        )

        for i in range(nearest_word_idx, max(0, nearest_word_idx - 10), -1):
            word_text = words[i]["word"].strip()
            if word_text and word_text[0].isupper():
                return words[i]

        return words[max(0, nearest_word_idx - 2)]

    def _find_nearest_word_end(
        self, target_time: float, words: list[dict]
    ) -> dict | None:
        """Находит конец слова ближайший к целевому времени."""
        if not words:
            return None

        nearest_word_idx = min(
            range(len(words)), key=lambda i: abs(words[i]["end"] - target_time)
        )

        return words[min(len(words) - 1, nearest_word_idx + 1)]

    def _filter_overlapping(self, segments: list[ViralSegment]) -> list[ViralSegment]:
        """Убирает пересекающиеся сегменты, оставляя с более высоким score."""
        if not segments:
            return segments

        # Сортируем по virality_score (убывание)
        sorted_segments = sorted(segments, key=lambda s: s.virality_score, reverse=True)

        filtered = []
        for seg in sorted_segments:
            # Проверяем пересечения с уже добавленными
            has_overlap = False
            for existing in filtered:
                if not (seg.end <= existing.start or seg.start >= existing.end):
                    has_overlap = True
                    logger.debug(
                        f"Segment overlap detected: "
                        f"[{seg.start:.1f}-{seg.end:.1f}] vs [{existing.start:.1f}-{existing.end:.1f}], "
                        f"keeping higher score ({existing.virality_score} > {seg.virality_score})"
                    )
                    break

            if not has_overlap:
                filtered.append(seg)

        # Сортируем обратно по времени
        filtered.sort(key=lambda s: s.start)

        logger.info(
            f"Filtered {len(sorted_segments) - len(filtered)} overlapping segments"
        )
        return filtered

    def analyze(self, transcript_path: Path) -> list[ViralSegment]:
        """
        Анализирует транскрипцию и находит вирусные фрагменты.

        Args:
            transcript_path: Путь к JSON файлу транскрипции

        Returns:
            Список найденных вирусных сегментов
        """
        logger.info(f"Analyzing transcript for viral segments: {transcript_path}")

        # Загружаем транскрипт
        transcript = VideoTranscriber.load_transcript(transcript_path)
        logger.info(
            f"Loaded transcript: {len(transcript.segments)} segments, "
            f"{transcript.duration:.1f}s ({transcript.duration / 60:.1f} min)"
        )

        # Форматируем для Claude
        transcript_dict = transcript.to_dict()
        formatted_text = format_transcript_for_analysis(transcript_dict["segments"])

        # Создаём промпт
        prompt = create_viral_segments_prompt(formatted_text)

        # Вызываем Claude
        logger.info("Sending request to Claude API...")
        response = self._call_claude_api(prompt)

        # Парсим результат
        segments = self._parse_segments(response)
        logger.info(f"Parsed {len(segments)} valid segments from Claude response")

        # Выравниваем по границам слов
        segments = self._align_to_word_boundaries(segments, transcript_dict["segments"])
        logger.info(f"Aligned {len(segments)} segments to word boundaries")

        # Фильтруем пересечения
        segments = self._filter_overlapping(segments)

        # Сортируем по virality_score
        segments.sort(key=lambda s: s.virality_score, reverse=True)

        logger.success(
            f"Analysis complete: {len(segments)} viral segments found "
            f"(avg score: {sum(s.virality_score for s in segments) / len(segments):.1f})"
            if segments
            else "Analysis complete: No viral segments found"
        )

        # Логируем топ-3
        for i, seg in enumerate(segments[:3], 1):
            logger.info(
                f"#{i} [{seg.start:.1f}s-{seg.end:.1f}s] "
                f"Score: {seg.virality_score}/10, "
                f"Hook: '{seg.hook}'"
            )

        return segments

    def save_analysis(self, segments: list[ViralSegment], output_path: Path):
        """Сохраняет результаты анализа в JSON."""
        data = {
            "total_segments": len(segments),
            "segments": [asdict(seg) for seg in segments],
        }

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        logger.info(f"Analysis saved to: {output_path}")

    @staticmethod
    def load_analysis(analysis_path: Path) -> list[ViralSegment]:
        """Загружает результаты анализа из JSON файла."""
        logger.info(f"Loading analysis from: {analysis_path}")

        with open(analysis_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        segments = []
        for seg_data in data.get("segments", []):
            segment = ViralSegment(
                start=float(seg_data["start"]),
                end=float(seg_data["end"]),
                hook=seg_data["hook"],
                punchline=seg_data["punchline"],
                virality_score=int(seg_data["virality_score"]),
                reason=seg_data["reason"],
                emotion=seg_data["emotion"],
            )
            segments.append(segment)

        logger.success(f"Loaded {len(segments)} segments from analysis")
        return segments
