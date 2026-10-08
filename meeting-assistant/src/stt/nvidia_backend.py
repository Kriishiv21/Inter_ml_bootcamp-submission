"""Whisper large-v3 hosted on build.nvidia.com (Riva gRPC)."""
from src import config


def transcribe(wav_path: str, hint: str = ""):  # hint is unused by this endpoint
    import riva.client
    if not config.NVIDIA_API_KEY:
        raise RuntimeError("NVIDIA_API_KEY is missing in .env")
    auth = riva.client.Auth(
        uri=config.NVIDIA_URI, use_ssl=True,
        metadata_args=[["function-id", config.NVIDIA_WHISPER_FUNCTION_ID],
                       ["authorization", f"Bearer {config.NVIDIA_API_KEY}"]])
    asr = riva.client.ASRService(auth)
    cfg = riva.client.RecognitionConfig(
        language_code="en", max_alternatives=1,
        enable_automatic_punctuation=True, enable_word_time_offsets=True)
    riva.client.add_audio_file_specs_to_config(cfg, wav_path)
    with open(wav_path, "rb") as f:
        resp = asr.offline_recognize(f.read(), cfg)
    segs = []
    for r in resp.results:
        if not r.alternatives:
            continue
        alt = r.alternatives[0]
        text = alt.transcript.strip()
        if not text:
            continue
        words = list(alt.words)
        segs.append({"start": round(words[0].start_time / 1000, 2) if words else 0.0,
                     "end": round(words[-1].end_time / 1000, 2) if words else 0.0,
                     "text": text})
    return segs
