import argparse
import json
import sys
from pathlib import Path

# Fix paths so it can import from src
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
ROOT = Path(__file__).resolve().parents[1]

from src.audio import to_wav16k
from src.stt import get_backend
from src.refine.refiner import refine_transcript
from src.document.extractor import (
    format_transcript_with_ids, 
    extract_meeting_record, 
    verify_and_filter_record, 
    generate_markdown
)

def main():
    ap = argparse.ArgumentParser(description="End-to-End Meeting Assistant Pipeline")
    ap.add_argument("--audio", required=True, help="Path to meeting audio file (mp3, wav, etc.)")
    ap.add_argument("--backend", default="nvidia", choices=["nvidia", "groq"], help="STT backend to use")
    ap.add_argument("--out", default="outputs", help="Output directory")
    a = ap.parse_args()

    audio_path = Path(a.audio)
    if not audio_path.exists():
        sys.exit(f"Error: Audio file not found at {audio_path}")

    outdir = Path(a.out) / audio_path.stem
    outdir.mkdir(parents=True, exist_ok=True)
    
    wav_16k = outdir / "input_16k.wav"

    print("=== STEP 1: VALIDATION & AUDIO PREPROCESSING ===")
    try:
        duration = to_wav16k(audio_path, wav_16k)
        print(f"-> Audio validated and converted. Duration: {duration:.1f}s")
    except Exception as e:
        sys.exit(f"Audio processing error: {e}")

    print("\n=== STEP 2: SPEECH-TO-TEXT (STT) ===")
    print(f"-> Running STT via backend: {a.backend}...")
    try:
        backend_result = get_backend(a.backend)(str(wav_16k), hint="")
        
        # Handle different return formats from backends safely
        if isinstance(backend_result, tuple):
            segments, raw_full_text = backend_result
        else:
            segments = backend_result
            raw_full_text = " ".join(s.get("text", "") for s in segments)
        
        # Inject stable Utterance IDs (U001, U002...) for evidence grounding
        valid_uids = set()
        for i, seg in enumerate(segments):
            uid = f"U{i+1:03d}"
            seg["id"] = uid
            valid_uids.add(uid)
            
        print(f"-> STT Complete. Generated {len(segments)} segments.")
    except Exception as e:
        sys.exit(f"STT Failure: {e}")

    # Save Raw Transcript Artifact
    raw_output = {
        "audio_seconds": round(duration, 1),
        "segments": segments,
        "full_text": raw_full_text
    }
    (outdir / "raw_transcript.json").write_text(json.dumps(raw_output, indent=2))

    print("\n=== STEP 3: EVIDENCE-DRIVEN REFINEMENT ===")
    print("-> Running Global Topic Pass, Span Detection, and Guardrails...")
    refined_segments = refine_transcript(segments)

    refined_full_text = " ".join(
        seg.get("text", "").strip()
        for seg in refined_segments
        if seg.get("text", "").strip()
    )

    (outdir / "refined_transcript.txt").write_text(refined_full_text)

    # Save the refined transcript with the same U-IDs as the raw transcript.
    refined_transcript_with_ids = format_transcript_with_ids(refined_segments)
    (outdir / "refined_transcript_with_ids.txt").write_text(
        refined_transcript_with_ids
    )
    refined_output = {
    "audio_seconds": round(duration, 1),
    "segments": refined_segments,
    "full_text": refined_full_text
    }

    (outdir / "refined_transcript.json").write_text(json.dumps(refined_output, indent=2))
    print("-> Refinement Complete.")

    print("\n=== STEP 4: EXTRACTION & QUALITY GATE ===")
    # Format transcript with U-IDs so the LLM can reference them
    formatted_transcript = format_transcript_with_ids(refined_segments)
    
    # Extract decisions and action items
    raw_record = extract_meeting_record(formatted_transcript)
    if not raw_record:
        sys.exit("Extraction failed to return a valid meeting record.")

    # Run the Verification Quality Gate (drops unverified items)
    verified_record = verify_and_filter_record(raw_record, refined_segments)

    print("\n=== STEP 5: RENDERING OUTPUTS ===")
    # 1. Save machine-readable JSON
    record_json_path = outdir / "record.json"
    record_dict = verified_record.model_dump()
    record_json_path.write_text(json.dumps(record_dict, indent=2))
    print(f"-> Saved: {record_json_path}")

    # 2. Render human-readable Markdown MOM from the EXACT same JSON
    record_md_path = outdir / "record.md"
    markdown_content = generate_markdown(verified_record)
    record_md_path.write_text(markdown_content)
    print(f"-> Saved: {record_md_path}")

    print(f"\nSUCCESS! All outputs successfully generated in: {outdir}/")

if __name__ == "__main__":
    main()