"""
Drishti Health — Vosk Offline Speech-to-Text

Fully offline STT for Kannada and Hindi using Vosk.
Works without internet — essential for rural deployment.

Models (download before hackathon):
- Hindi: https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip (~42MB)
- Kannada: Use Hindi model with transliteration (Kannada model pending)
- English: https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip (~40MB)

Usage:
    vosk = VoskOfflineSTT(language="hi")
    text = vosk.transcribe(audio_bytes)
"""

import os
import json
import wave
import io
from pathlib import Path
from typing import Optional

# Conditional import — install vosk: pip install vosk
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    VOSK_AVAILABLE = False


# Model download URLs
VOSK_MODELS = {
    "en": {
        "name": "vosk-model-small-en-us-0.15",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-en-us-0.15.zip",
        "size": "40MB",
    },
    "hi": {
        "name": "vosk-model-small-hi-0.22",
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
        "size": "42MB",
    },
    "kn": {
        "name": "vosk-model-small-hi-0.22",  # Using Hindi model as fallback
        "url": "https://alphacephei.com/vosk/models/vosk-model-small-hi-0.22.zip",
        "size": "42MB",
        "note": "Using Hindi model — Kannada model pending from Vosk"
    },
}


class VoskOfflineSTT:
    """Offline speech-to-text using Vosk."""

    def __init__(self, language: str = "hi", model_dir: str = "voice/models"):
        self.language = language
        self.model_dir = Path(model_dir)
        self.model = None
        self.available = False

        if not VOSK_AVAILABLE:
            print("⚠️  Vosk not installed. Run: pip install vosk")
            return

        self._load_model()

    def _load_model(self):
        """Load Vosk model for the specified language."""
        model_info = VOSK_MODELS.get(self.language, VOSK_MODELS["hi"])
        model_path = self.model_dir / model_info["name"]

        if model_path.exists():
            try:
                self.model = Model(str(model_path))
                self.available = True
                print(f"✅ Vosk {self.language} model loaded from {model_path}")
            except Exception as e:
                print(f"⚠️  Failed to load Vosk model: {e}")
        else:
            print(f"⚠️  Vosk model not found at {model_path}")
            print(f"   Download from: {model_info['url']}")
            print(f"   Extract to: {model_path}")

    def transcribe(self, audio_bytes: bytes, sample_rate: int = 16000) -> str:
        """
        Transcribe audio bytes to text.

        Args:
            audio_bytes: Raw audio data (WAV format preferred)
            sample_rate: Audio sample rate (default 16000 Hz)

        Returns:
            Transcribed text string
        """
        if not self.available:
            return "(Vosk model not available — download model first)"

        rec = KaldiRecognizer(self.model, sample_rate)

        # Try to read as WAV first
        try:
            wf = wave.open(io.BytesIO(audio_bytes), "rb")
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                rec.AcceptWaveform(data)
        except Exception:
            # Raw audio fallback
            chunk_size = 4000
            for i in range(0, len(audio_bytes), chunk_size):
                rec.AcceptWaveform(audio_bytes[i:i + chunk_size])

        result = json.loads(rec.FinalResult())
        return result.get("text", "")

    def transcribe_file(self, audio_path: str) -> str:
        """Transcribe audio from file path."""
        with open(audio_path, "rb") as f:
            return self.transcribe(f.read())

    @staticmethod
    def download_model(language: str = "hi", model_dir: str = "voice/models"):
        """Download Vosk model for the specified language."""
        import urllib.request
        import zipfile

        model_info = VOSK_MODELS.get(language, VOSK_MODELS["hi"])
        model_path = Path(model_dir)
        model_path.mkdir(parents=True, exist_ok=True)

        zip_path = model_path / f"{model_info['name']}.zip"

        if (model_path / model_info["name"]).exists():
            print(f"✅ Model already exists: {model_path / model_info['name']}")
            return

        print(f"📥 Downloading Vosk {language} model ({model_info['size']})...")
        print(f"   URL: {model_info['url']}")

        try:
            urllib.request.urlretrieve(model_info["url"], zip_path)
            print("📦 Extracting...")
            with zipfile.ZipFile(zip_path, "r") as zip_ref:
                zip_ref.extractall(model_path)
            os.remove(zip_path)
            print(f"✅ Model ready at: {model_path / model_info['name']}")
        except Exception as e:
            print(f"❌ Download failed: {e}")


if __name__ == "__main__":
    if not VOSK_AVAILABLE:
        print("Install vosk: pip install vosk")
    else:
        print("Available models:")
        for lang, info in VOSK_MODELS.items():
            print(f"  {lang}: {info['name']} ({info['size']})")
            if "note" in info:
                print(f"       Note: {info['note']}")
