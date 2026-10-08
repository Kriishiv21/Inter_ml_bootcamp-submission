"""Validate an upload and convert it to 16 kHz mono wav."""
import subprocess
from pathlib import Path
import soundfile as sf
from src.config import SAMPLE_RATE


def to_wav16k(src: Path, dst: Path) -> float:
    """Returns duration in seconds, or raises ValueError with a readable message."""
    src, dst = Path(src), Path(dst)
    if not src.exists() or src.stat().st_size == 0:
        raise ValueError("File is missing or empty.")
    try:
        r = subprocess.run(
            ["ffmpeg", "-y", "-i", str(src), "-ac", "1", "-ar", str(SAMPLE_RATE),
             "-af", "loudnorm", str(dst)], capture_output=True, text=True)
    except FileNotFoundError:
        raise ValueError("ffmpeg is not installed or not on PATH.")
    if r.returncode != 0 or not dst.exists() or dst.stat().st_size < 1000:
        raise ValueError("Unsupported or unreadable audio file.")
    dur = sf.info(str(dst)).duration
    if dur < 0.5:
        raise ValueError("Audio is too short or silent.")
    return dur
