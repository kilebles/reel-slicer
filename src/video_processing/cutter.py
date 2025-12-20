"""Video cutting module for segmenting videos based on analysis results."""

import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import ffmpeg

from src.logger import logger
from src.settings import settings


class VideoCutter:
    """Cuts video into segments based on analysis results."""

    def __init__(self, video_path: str, analysis_path: str, output_dir: str = "data/output"):
        """
        Initialize VideoCutter.

        Args:
            video_path: Path to input video file
            analysis_path: Path to analysis.json file with segment timestamps
            output_dir: Directory to save output video segments
        """
        self.video_path = Path(video_path)
        self.analysis_path = Path(analysis_path)
        self.output_dir = Path(output_dir)

        # Validate input files
        if not self.video_path.exists():
            raise FileNotFoundError(f"Video file not found: {self.video_path}")
        if not self.analysis_path.exists():
            raise FileNotFoundError(f"Analysis file not found: {self.analysis_path}")

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.segments: List[Dict[str, Any]] = []
        logger.info(f"VideoCutter initialized: video={self.video_path}, analysis={self.analysis_path}")

    def load_analysis(self) -> None:
        """Load segment data from analysis.json."""
        try:
            with open(self.analysis_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            self.segments = data.get('segments', [])
            total = data.get('total_segments', len(self.segments))

            logger.success(f"Loaded {len(self.segments)} segments from analysis (total: {total})")

            if not self.segments:
                logger.warning("No segments found in analysis file")

        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse analysis JSON: {e}")
            raise
        except Exception as e:
            logger.error(f"Error loading analysis: {e}")
            raise

    def cut_segment(
        self,
        start: float,
        end: float,
        output_path: str,
        **metadata
    ) -> bool:
        """
        Cut a single segment from video using ffmpeg.

        Args:
            start: Start time in seconds
            end: End time in seconds
            output_path: Path for output video file
            **metadata: Additional segment metadata (hook, punchline, etc.)

        Returns:
            True if successful, False otherwise
        """
        try:
            start_offset = settings.SEGMENT_START_OFFSET
            adjusted_start = start + start_offset
            adjusted_duration = end - adjusted_start

            logger.info(f"Cutting segment: {start:.1f}s - {end:.1f}s (duration: {end - start:.1f}s)")
            logger.debug(f"  Adjusted start: {adjusted_start:.2f}s (offset: +{start_offset}s)")
            if 'hook' in metadata:
                logger.debug(f"  Hook: {metadata['hook'][:50]}...")

            # Use ffmpeg to cut the segment
            # -ss: start time, -t: duration, -c copy: copy without re-encoding
            (
                ffmpeg
                .input(str(self.video_path), ss=adjusted_start, t=adjusted_duration)
                .output(
                    str(output_path),
                    c='copy',  # Copy codec without re-encoding (fast)
                    avoid_negative_ts='make_zero'  # Fix timestamp issues
                )
                .overwrite_output()
                .run(capture_stdout=True, capture_stderr=True, quiet=True)
            )

            logger.success(f"Segment saved: {output_path}")
            return True

        except ffmpeg.Error as e:
            logger.error(f"FFmpeg error cutting segment {start}-{end}: {e.stderr.decode()}")
            return False
        except Exception as e:
            logger.error(f"Unexpected error cutting segment {start}-{end}: {e}")
            return False

    def cut_all_segments(self, name_pattern: str = "segment_{index:02d}.mp4") -> List[Path]:
        """
        Cut all segments from the loaded analysis.

        Args:
            name_pattern: Output filename pattern. Use {index} for segment number,
                         {emotion}, {virality_score}, etc. for metadata fields

        Returns:
            List of paths to created video segments
        """
        if not self.segments:
            logger.warning("No segments to cut. Call load_analysis() first.")
            return []

        logger.info(f"Starting to cut {len(self.segments)} segments...")

        output_files = []
        success_count = 0

        for index, segment in enumerate(self.segments, start=1):
            start = segment.get('start')
            end = segment.get('end')

            if start is None or end is None:
                logger.warning(f"Segment {index} missing start/end times, skipping")
                continue

            # Format output filename with segment data
            format_data = {'index': index, **segment}
            filename = name_pattern.format(**format_data)

            output_path = self.output_dir / filename

            # Cut the segment
            metadata = {k: v for k, v in segment.items() if k not in ('start', 'end')}
            if self.cut_segment(start, end, str(output_path), **metadata):
                output_files.append(output_path)
                success_count += 1

        logger.success(f"Completed: {success_count}/{len(self.segments)} segments cut successfully")
        return output_files

    def cut_segment_with_metadata(
        self,
        segment_index: int,
        include_metadata: bool = False
    ) -> Optional[Path]:
        """
        Cut a specific segment by index with optional metadata overlay.

        Args:
            segment_index: Index of segment to cut (0-based)
            include_metadata: If True, overlay text with hook/punchline (requires re-encoding)

        Returns:
            Path to output file if successful, None otherwise
        """
        if segment_index < 0 or segment_index >= len(self.segments):
            logger.error(f"Invalid segment index: {segment_index}")
            return None

        segment = self.segments[segment_index]
        start = segment.get('start')
        end = segment.get('end')

        if start is None or end is None:
            logger.error(f"Segment {segment_index} missing timestamps")
            return None

        filename = f"segment_{segment_index:02d}_{segment.get('emotion', 'unknown')}.mp4"
        output_path = self.output_dir / filename

        if include_metadata:
            # TODO: Implement text overlay with hook/punchline
            logger.warning("Metadata overlay not yet implemented, cutting without overlay")

        metadata = {k: v for k, v in segment.items() if k not in ('start', 'end')}
        success = self.cut_segment(start, end, str(output_path), **metadata)
        return output_path if success else None


def cut_video_from_analysis(
    video_path: str = "data/video.mp4",
    analysis_path: str = "data/analysis.json",
    output_dir: str = "data/output"
) -> List[Path]:
    """
    Convenience function to cut video segments from analysis file.

    Args:
        video_path: Path to input video
        analysis_path: Path to analysis JSON
        output_dir: Directory for output segments

    Returns:
        List of paths to created segments
    """
    cutter = VideoCutter(video_path, analysis_path, output_dir)
    cutter.load_analysis()
    return cutter.cut_all_segments()


if __name__ == "__main__":
    # Example usage
    logger.info("Starting video cutting process...")

    output_files = cut_video_from_analysis(
        video_path="data/video.mp4",
        analysis_path="data/analysis.json",
        output_dir="data/output"
    )

    logger.info(f"Created {len(output_files)} video segments:")
    for file in output_files:
        logger.info(f"  - {file}")
