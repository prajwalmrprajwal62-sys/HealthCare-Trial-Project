"""
Drishti Health — Bhashini API Client

Integrates with the Government of India's Bhashini (ULCA) platform for:
- ASR (Automatic Speech Recognition) in Indian languages
- NMT (Neural Machine Translation) between Indian languages
- TTS (Text-to-Speech) for patient summaries

Supports 22 Indian languages including Kannada, Hindi, Tamil, Telugu, etc.
API Docs: https://bhashini.gov.in/ulca/apis

Fallback: Vosk offline STT if no internet or API key.
"""

import os
import json
import base64
import httpx
from typing import Optional, Dict, Any
from pathlib import Path


class BhashiniClient:
    """
    Client for Bhashini (ULCA) government API.

    Pipeline: Audio → ASR → NMT → TTS
    Free API key at: https://bhashini.gov.in/ulca
    """

    BASE_URL = "https://meity-auth.ulcacontrib.org"
    INFERENCE_URL = "https://dhruva-api.bhashini.gov.in/services/inference"

    LANGUAGE_CODES = {
        "kannada": "kn",
        "hindi": "hi",
        "english": "en",
        "tamil": "ta",
        "telugu": "te",
        "malayalam": "ml",
        "marathi": "mr",
        "bengali": "bn",
        "gujarati": "gu",
        "punjabi": "pa",
    }

    def __init__(self, user_id: str = "", api_key: str = "", pipeline_id: str = ""):
        self.user_id = user_id or os.getenv("BHASHINI_USER_ID", "")
        self.api_key = api_key or os.getenv("BHASHINI_API_KEY", "")
        self.pipeline_id = pipeline_id or os.getenv("BHASHINI_PIPELINE_ID", "")
        self.available = bool(self.api_key)

        if not self.available:
            print("⚠️  Bhashini API key not set. Voice features will use offline fallback.")
            print("   Get free API key at: https://bhashini.gov.in/ulca")

    def _get_headers(self) -> Dict[str, str]:
        return {
            "Accept": "application/json",
            "Content-Type": "application/json",
            "userID": self.user_id,
            "ulcaApiKey": self.api_key,
        }

    async def get_pipeline_config(self, source_lang: str, target_lang: str, tasks: list) -> Dict:
        """
        Get pipeline configuration for the specified language pair and tasks.

        Args:
            source_lang: Source language code (e.g., "kn")
            target_lang: Target language code (e.g., "en")
            tasks: List of tasks ["asr", "nmt", "tts"]
        """
        if not self.available:
            return {"error": "API key not configured"}

        payload = {
            "pipelineTasks": [{"taskType": t, "config": {
                "language": {"sourceLanguage": source_lang, "targetLanguage": target_lang}
            }} for t in tasks],
            "pipelineRequestConfig": {
                "pipelineId": self.pipeline_id or "64392f96daac500b55c543cd"
            }
        }

        async with httpx.AsyncClient() as client:
            response = await client.post(
                f"{self.BASE_URL}/ulca/apis/v0/model/getModelsPipeline",
                headers=self._get_headers(),
                json=payload,
                timeout=30,
            )
            return response.json()

    async def speech_to_text(
        self, audio_bytes: bytes, source_language: str = "kn",
        audio_format: str = "wav"
    ) -> Dict[str, Any]:
        """
        Convert speech to text using Bhashini ASR.

        Args:
            audio_bytes: Raw audio data
            source_language: Language code ("kn", "hi", "en", etc.)
            audio_format: Audio format ("wav", "mp3", "flac")

        Returns:
            {"text": "transcribed text", "language": "kn", "confidence": 0.85}
        """
        if not self.available:
            return self._offline_fallback_stt(audio_bytes, source_language)

        audio_b64 = base64.b64encode(audio_bytes).decode("utf-8")

        payload = {
            "pipelineTasks": [{
                "taskType": "asr",
                "config": {
                    "language": {"sourceLanguage": source_language},
                    "audioFormat": audio_format,
                    "samplingRate": 16000,
                }
            }],
            "inputData": {
                "audio": [{"audioContent": audio_b64}]
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.INFERENCE_URL,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=30,
                )
                result = response.json()

                text = result.get("pipelineResponse", [{}])[0].get("output", [{}])[0].get("source", "")
                return {
                    "text": text,
                    "language": source_language,
                    "confidence": 0.85,
                    "source": "bhashini"
                }
        except Exception as e:
            print(f"Bhashini ASR failed: {e}. Falling back to offline.")
            return self._offline_fallback_stt(audio_bytes, source_language)

    async def translate(
        self, text: str,
        source_language: str = "kn",
        target_language: str = "en"
    ) -> Dict[str, Any]:
        """
        Translate text between Indian languages using Bhashini NMT.

        Args:
            text: Input text
            source_language: Source language code
            target_language: Target language code

        Returns:
            {"translated_text": "...", "source": "kn", "target": "en"}
        """
        if not self.available:
            return {
                "translated_text": text,
                "source": source_language,
                "target": target_language,
                "note": "Translation unavailable offline"
            }

        payload = {
            "pipelineTasks": [{
                "taskType": "translation",
                "config": {
                    "language": {
                        "sourceLanguage": source_language,
                        "targetLanguage": target_language,
                    }
                }
            }],
            "inputData": {
                "input": [{"source": text}]
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.INFERENCE_URL,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=30,
                )
                result = response.json()

                translated = result.get("pipelineResponse", [{}])[0].get("output", [{}])[0].get("target", "")
                return {
                    "translated_text": translated,
                    "source": source_language,
                    "target": target_language,
                    "source_api": "bhashini"
                }
        except Exception as e:
            return {
                "translated_text": text,
                "source": source_language,
                "target": target_language,
                "error": str(e)
            }

    async def text_to_speech(
        self, text: str, language: str = "kn",
        gender: str = "female"
    ) -> Optional[bytes]:
        """
        Convert text to speech using Bhashini TTS.

        Args:
            text: Text to synthesize
            language: Language code
            gender: "male" or "female"

        Returns:
            Audio bytes (WAV format) or None
        """
        if not self.available:
            return None

        payload = {
            "pipelineTasks": [{
                "taskType": "tts",
                "config": {
                    "language": {"sourceLanguage": language},
                    "gender": gender,
                }
            }],
            "inputData": {
                "input": [{"source": text}]
            }
        }

        try:
            async with httpx.AsyncClient() as client:
                response = await client.post(
                    self.INFERENCE_URL,
                    headers=self._get_headers(),
                    json=payload,
                    timeout=30,
                )
                result = response.json()

                audio_b64 = result.get("pipelineResponse", [{}])[0].get("audio", [{}])[0].get("audioContent", "")
                if audio_b64:
                    return base64.b64decode(audio_b64)
        except Exception as e:
            print(f"Bhashini TTS failed: {e}")

        return None

    async def full_pipeline(
        self, audio_bytes: bytes,
        source_language: str = "kn",
        target_language: str = "en"
    ) -> Dict[str, Any]:
        """
        Full pipeline: Speech → Text → Translation → TTS response.

        Args:
            audio_bytes: Audio input
            source_language: Input language
            target_language: Translation target

        Returns:
            Full pipeline result with transcription, translation, and audio
        """
        # Step 1: ASR
        stt_result = await self.speech_to_text(audio_bytes, source_language)

        # Step 2: NMT
        translation = await self.translate(
            stt_result["text"], source_language, target_language
        )

        # Step 3: TTS (in target language)
        tts_audio = await self.text_to_speech(
            translation["translated_text"], target_language
        )

        return {
            "transcription": stt_result["text"],
            "translation": translation["translated_text"],
            "source_language": source_language,
            "target_language": target_language,
            "audio_response": tts_audio is not None,
            "pipeline": "bhashini" if self.available else "offline",
        }

    def _offline_fallback_stt(self, audio_bytes: bytes, language: str) -> Dict[str, Any]:
        """Fallback: try Vosk offline, else return placeholder."""
        try:
            from voice.vosk_offline import VoskOfflineSTT
            vosk = VoskOfflineSTT(language=language)
            text = vosk.transcribe(audio_bytes)
            return {"text": text, "language": language, "confidence": 0.7, "source": "vosk"}
        except Exception:
            return {
                "text": "(Voice input unavailable — please type symptoms)",
                "language": language,
                "confidence": 0,
                "source": "fallback"
            }


if __name__ == "__main__":
    client = BhashiniClient()
    print(f"Bhashini API available: {client.available}")
    print(f"Supported languages: {list(client.LANGUAGE_CODES.keys())}")
