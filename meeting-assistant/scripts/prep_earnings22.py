"""Build a short test clip + ground truth from Earnings-22 (chunked subset).

  python scripts/prep_earnings22.py --file-id 4483589 --minutes 1

Writes:
  data/raw/earnings22_<id>_<N>min.wav                      audio to transcribe
  data/ground_truth/earnings22_<id>_<N>min.txt             verbatim reference (fillers kept)
  data/ground_truth/earnings22_<id>_<N>min_segments.json   per-segment text + times
"""
import argparse
import io
import json
from pathlib import Path

import numpy as np
import soundfile as sf
from datasets import Audio, load_dataset

ROOT = Path(__file__).resolve().parents[1]

ap = argparse.ArgumentParser()
ap.add_argument("--file-id", default="4483589")
ap.add_argument("--minutes", type=float, default=1.0)
a = ap.parse_args()

# stream so we don't download all 57k segments
ds = load_dataset("distil-whisper/earnings22", "chunked", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(decode=False))  # raw bytes, avoids extra decoder installs

rows, total, sr = [], 0.0, None
for r in ds:
    if str(r["file_id"]) != a.file_id:
        if rows:
            break          # rows of one call are grouped, so stop once we leave it
        continue
    wav, sr = sf.read(io.BytesIO(r["audio"]["bytes"]), dtype="float32")
    if wav.ndim > 1:
        wav = wav.mean(axis=1)
    rows.append({"start": float(r["start_ts"]), "end": float(r["end_ts"]),
                 "text": r["transcription"].strip(), "audio": wav})
    total += len(wav) / sr
    if total >= a.minutes * 60:
        break
if not rows:
    raise SystemExit(f"No rows found for file_id {a.file_id}")

rows.sort(key=lambda x: x["start"])
gap = np.zeros(int(0.4 * sr), dtype="float32")
audio = np.concatenate([x for r in rows for x in (r["audio"], gap)])

name = f"earnings22_{a.file_id}_{a.minutes:g}min"
(ROOT / "data/raw").mkdir(parents=True, exist_ok=True)
(ROOT / "data/ground_truth").mkdir(parents=True, exist_ok=True)
sf.write(ROOT / f"data/raw/{name}.wav", audio, sr)
(ROOT / f"data/ground_truth/{name}.txt").write_text(" ".join(r["text"] for r in rows))
(ROOT / f"data/ground_truth/{name}_segments.json").write_text(json.dumps(
    [{k: v for k, v in r.items() if k != "audio"} for r in rows], indent=2))
print(f"{name}: {len(rows)} segments, {len(audio) / sr:.0f}s")