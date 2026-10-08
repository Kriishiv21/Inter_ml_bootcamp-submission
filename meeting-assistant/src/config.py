"""Central settings. Secrets come from .env, never hardcode them."""
import os
from pathlib import Path
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")

NVIDIA_API_KEY = os.getenv("NVIDIA_API_KEY")
NVIDIA_URI = "grpc.nvcf.nvidia.com:443"
NVIDIA_WHISPER_FUNCTION_ID = "b702f636-f60c-4a3d-a6f4-f3568c13bd7d"  
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"
REFINER_MODEL = os.getenv("REFINER_MODEL", "meta-llama/llama-3.3-70b-instruct")
DOCUMENTER_MODEL = os.getenv("DOCUMENTER_MODEL", "openai/gpt-oss-120b")
SAMPLE_RATE = 16000
