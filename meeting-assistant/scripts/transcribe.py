"""
Run one or more Whisper backends on an audio file and save Transcript JSON.

  python scripts/transcribe.py --audio data/raw/clip.wav --ref data/raw/clip.txt \
      --backends groq nvidia faster_whisper --hint "Kubernetes, CI/CD" --terms data/raw/terms.txt

Output: outputs/<audio name>/<backend>.json and results.csv
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]
from src.audio import to_wav16k          # noqa: E402
from src.metrics import (wer, wer_verbatim, term_recall, filler_count,  # noqa: E402
                         alignment_text)  # noqa: E402
from src.stt import BACKENDS, get_backend  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--audio", required=True)
    ap.add_argument("--ref", help="reference transcript .txt")
    ap.add_argument("--backends", nargs="+", default=["groq"], choices=BACKENDS)
    ap.add_argument("--hint", default="", help="comma-separated domain terms")
    ap.add_argument("--terms", help="file with one domain term per line")
    ap.add_argument("--out", default="outputs")
    a = ap.parse_args()

    src = Path(a.audio)
    outdir = Path(a.out) / src.stem
    outdir.mkdir(parents=True, exist_ok=True)
    wav = outdir / "input_16k.wav"
    try:
        dur = to_wav16k(src, wav)
    except ValueError as e:
        sys.exit(f"Audio error: {e}")

    ref_path = Path(a.ref) if a.ref else ROOT / "data" / "ground_truth" / f"{src.stem}.txt"
    ref = ref_path.read_text() if ref_path.exists() else None
    terms = ([t.strip() for t in Path(a.terms).read_text().splitlines() if t.strip()]
             if a.terms else [])
    rows = []
    for name in a.backends:
        print(f"[{name}] running ...")
        t0 = time.time()
        try:
            segs = get_backend(name)(str(wav), a.hint)
            
            # NEW: Inject stable Utterance IDs for evidence tracking
            for i, seg in enumerate(segs):
                seg["id"] = f"U{i+1:03d}"  # Generates U001, U002, etc.
                
        except Exception as ex:
            print(f"[{name}] FAILED: {ex}")
            rows.append({"backend": name, "error": str(ex)})
            continue
        rt = time.time() - t0
        full = " ".join(s["text"] for s in segs)
        (outdir / f"{name}.json").write_text(json.dumps({
            "backend": name, "segments": segs, "full_text": full,
            "audio_seconds": round(dur, 1), "runtime_s": round(rt, 1)}, indent=2))
        (outdir / f"{name}.txt").write_text(full)
        if ref:
            (outdir / f"{name}_alignment.txt").write_text(alignment_text(ref, full))
        row = {"backend": name, "audio_s": round(dur, 1), "runtime_s": round(rt, 1),
               "wer": round(wer(ref, full), 4) if ref else "",
               "wer_verbatim": round(wer_verbatim(ref, full), 4) if ref else "",
               "fillers_ref": filler_count(ref) if ref else "",
               "fillers_hyp": filler_count(full),
               "term_recall": round(term_recall(terms, full), 3) if terms else ""}
        rows.append(row)
        print(row)

    keys = sorted({k for r in rows for k in r}, key=lambda k: k != "backend")
    with open(outdir / "results.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    print(f"Saved to {outdir}/")


if __name__ == "__main__":
    main()
