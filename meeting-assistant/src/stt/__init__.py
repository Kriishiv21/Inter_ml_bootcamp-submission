"""Backend registry. Every backend: transcribe(wav_path, hint) -> [{start,end,text}]."""
BACKENDS = ("nvidia",)


def get_backend(name: str):
    # lazy imports so a missing package only breaks the backend that needs it
    if name == "nvidia":
        from .nvidia_backend import transcribe
    else:
        raise ValueError(f"Unknown backend '{name}'. Choose from {BACKENDS}.")
    return transcribe