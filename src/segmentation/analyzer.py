"""Анализ транскрипций для интеллектуальной сегментации видео с помощью Claude AI."""

import json
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

from anthropic import Anthropic, RateLimitError

from src.logger import logger
from src.segmentation.prompts import create_segmentation_prompt
from src.settings import settings
from src.transcription.transcriber import VideoTranscriber


class TranscriptAnalyzer:
    """Анализирует транскрипции с помощью Claude AI для определения оптимальных границ сегментов."""

    def __init__(self):
        """Инициализирует analyzer с клиентом Anthropic."""
        self.client = Anthropic(api_key=settings.API_KEY)
        self.model = settings.ANTHROPIC_MODEL
        self.max_tokens = settings.MAX_TOKENS
        self.chunk_size = settings.CHUNK_SIZE

    def _estimate_tokens(self, text: str) -> int:
        """
        Оценивает количество токенов в тексте.

        Примерная оценка: 1 token ≈ 4 символа для русского текста.
        Консервативная оценка, чтобы избежать превышения лимитов.

        Args:
            text: Текст для оценки токенов

        Returns:
            Примерное количество токенов
        """
        return len(text) // 3

    def _create_chunks(self, transcript) -> list[dict[str, Any]]:
        """
        Разбивает транскрипцию на фрагменты примерно по CHUNK_SIZE токенов.

        Каждый фрагмент сохраняет границы сегментов и включает информацию о времени.

        Args:
            transcript: Объект Transcript для разбиения

        Returns:
            Список словарей фрагментов с сегментами и метаданными
        """
        chunks = []
        current_chunk = []
        current_tokens = 0

        for segment in transcript.segments:
            segment_dict = asdict(segment)
            segment_text = segment_dict["text"]
            segment_tokens = self._estimate_tokens(segment_text)

            
            if current_tokens + segment_tokens > self.chunk_size and current_chunk:
                chunks.append({
                    "segments": current_chunk,
                    "token_count": current_tokens,
                    "start_time": current_chunk[0]["start"],
                    "end_time": current_chunk[-1]["end"]
                })
                current_chunk = []
                current_tokens = 0

            current_chunk.append(segment_dict)
            current_tokens += segment_tokens

        
        if current_chunk:
            chunks.append({
                "segments": current_chunk,
                "token_count": current_tokens,
                "start_time": current_chunk[0]["start"],
                "end_time": current_chunk[-1]["end"]
            })

        logger.info(f"Created {len(chunks)} chunks from transcript")
        for i, chunk in enumerate(chunks):
            logger.debug(
                f"Chunk {i + 1}: {chunk['token_count']} tokens, "
                f"time {chunk['start_time']:.1f}s - {chunk['end_time']:.1f}s"
            )

        return chunks

    def _create_analysis_prompt(self, chunk: dict[str, Any]) -> str:
        """
        Создает промпт для анализа фрагмента транскрипции с помощью Claude.

        Args:
            chunk: Словарь фрагмента с сегментами

        Returns:
            Отформатированный промпт
        """
        segments_json = json.dumps(chunk["segments"], ensure_ascii=False, indent=2)
        return create_segmentation_prompt(
            segments_json=segments_json,
            start_time=chunk['start_time'],
            end_time=chunk['end_time']
        )

    def _call_claude_api(self, prompt: str, max_retries: int = 3) -> dict[str, Any]:
        """
        Выполняет API-вызов к Claude для анализа с логикой повторных попыток.

        Args:
            prompt: Промпт для анализа
            max_retries: Максимальное количество повторных попыток при rate limit

        Returns:
            Распарсенный JSON-ответ от Claude
        """
        retry_delay = 60

        for attempt in range(max_retries):
            try:
                message = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "user", "content": prompt}
                    ]
                )

                response_text = message.content[0].text
                logger.debug(f"Claude response: {response_text[:200]}...")

            
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

    def analyze(self, transcript_path: Path) -> dict[str, Any]:
        """
        Анализирует транскрипцию и определяет границы сегментов.

        Args:
            transcript_path: Путь к JSON-файлу транскрипции

        Returns:
            Результаты анализа с определенными сегментами
        """
        logger.info(f"Analyzing transcript: {transcript_path}")

        
        transcript = VideoTranscriber.load_transcript(transcript_path)
        logger.info(f"Loaded transcript: {len(transcript.segments)} segments, {transcript.duration:.1f}s")

        
        chunks = self._create_chunks(transcript)

        
        all_segments = []
        for i, chunk in enumerate(chunks):
            logger.info(f"Analyzing chunk {i + 1}/{len(chunks)}...")

            prompt = self._create_analysis_prompt(chunk)
            result = self._call_claude_api(prompt)

            if "segments" in result:
                all_segments.extend(result["segments"])
                logger.success(f"Chunk {i + 1}: Found {len(result['segments'])} segments")
            else:
                logger.warning(f"Chunk {i + 1}: No segments found in response")

            
            if i < len(chunks) - 1:
                delay = 2  
                logger.info(f"Waiting {delay}s before next chunk...")
                time.sleep(delay)

        
        analysis_result = {
            "original_duration": transcript.duration,
            "original_segments": len(transcript.segments),
            "identified_segments": len(all_segments),
            "segments": all_segments
        }

        logger.success(f"Analysis complete: {len(all_segments)} segments identified")
        return analysis_result

    def save_analysis(self, analysis: dict[str, Any], output_path: Path) -> None:
        """
        Сохраняет результаты анализа в JSON-файл.

        Args:
            analysis: Результаты анализа
            output_path: Путь для сохранения результатов
        """
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(analysis, f, ensure_ascii=False, indent=2)

        logger.info(f"Analysis saved to: {output_path}")
