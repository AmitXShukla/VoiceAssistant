"""
config.py — All settings for Atlas Voice Assistant
Edit this file to customise your setup.

Python 3.12: dataclass fields with default values must come after
fields without defaults. tts_backend moved to correct position.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field


@dataclass
class Config:
    # ── Identity ──────────────────────────────────────────────────────────────
    assistant_name: str = "Atlas"
    wake_word: str = "hey atlas"

    # ── Ollama LLM ────────────────────────────────────────────────────────────
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "gpt-oss:120b"
    ollama_system_prompt: str = (
        "You are Atlas, a helpful and concise voice assistant running entirely locally. "
        "Keep responses SHORT (2-4 sentences) — they will be spoken aloud. "
        "Never use markdown, bullet points, emojis, or special characters. "
        "Speak naturally as if in conversation. "
        "When you want to use a tool, respond ONLY with a JSON object: "
        '{"tool": "tool_name", "args": {"key": "value"}} '
        "Available tools: web_search(query), get_datetime(), "
        "set_reminder(text, seconds), list_reminders(), cancel_reminder(id)."
    )
    ollama_temperature: float = 0.7
    ollama_max_tokens: int = 300
    context_window: int = 10  # conversation turns to keep

    # ── STT — faster-whisper ──────────────────────────────────────────────────
    # whisper_model: str = "large-v3"  # tiny | base | small | medium | large-v3
    # whisper_device: str = "cuda"
    # whisper_compute_type: str = "float16"  # float16 | int8 (int8 saves VRAM)
    whisper_model: str = "medium"  # tiny | base | small | medium | large-v3
    whisper_device: str = "cpu"
    whisper_compute_type: str = "int8"  # float16 | int8 (int8 saves VRAM)
    whisper_language: str | None = "en"  # None = auto-detect

    # ── TTS backend selection ─────────────────────────────────────────────────
    tts_backend: str = "xtts"  # "xtts" or "rvc" (set by CLI, overridable here)

    # ── TTS — Coqui XTTS-v2 ──────────────────────────────────────────────────
    xtts_voice_sample: str = "sample_en_amit.wav"
    xtts_model_name: str = "tts_models/multilingual/multi-dataset/xtts_v2"
    xtts_language: str = "en"
    # xtts_device: str = "cuda"
    xtts_device: str = "cpu"

    # ── TTS — RVC (alternative backend) ──────────────────────────────────────
    rvc_model_path: str = "rvc_models/my_voice.pth"
    rvc_index_path: str = "rvc_models/my_voice.index"
    rvc_pitch_shift: int = 0
    rvc_f0_method: str = "rmvpe"  # rmvpe | crepe | harvest | pm
    rvc_base_tts_voice: str = "en-US-JennyNeural"
    rvc_device: str = "cuda"

    # ── Wake Word — OpenWakeWord ──────────────────────────────────────────────
    wake_word_model_path: str = "wake_word/hey_atlas.onnx"
    wake_word_threshold: float = 0.7
    wake_word_fallback: str = "hey_jarvis"

    # ── Audio recording / playback ────────────────────────────────────────────
    samplerate: int = 16000
    channels: int = 1
    input_device: int | None = None  # None = system default
    output_device: int | None = None  # run --list-devices to find index
    # Silero VAD
    vad_threshold: float = 0.5
    vad_min_speech_ms: int = 250
    vad_min_silence_ms: int = 1200
    # Interrupt detection (while Atlas is speaking)
    interrupt_rms_threshold: float = 0.04
    interrupt_hold_ms: int = 350
    # Noise suppression
    noise_suppress: bool = True  # use noisereduce
    deepfilter: bool = False  # use DeepFilterNet (needs extra install)
    max_record_seconds: int = 30

    # ── Memory ────────────────────────────────────────────────────────────────
    db_path: str = "atlas_memory.db"

    # ── Tools ─────────────────────────────────────────────────────────────────
    search_max_results: int = 3
    serpapi_key: str = os.environ.get("SERPAPI_KEY", "")

    # ── Web UI ────────────────────────────────────────────────────────────────
    ui_port: int = 7860
    ui_host: str = "0.0.0.0"

    # ── Debug ─────────────────────────────────────────────────────────────────
    log_audio: bool = False
    audio_log_dir: str = "audio_logs"

    def validate(self) -> None:
        """Check critical config values on startup and raise with clear messages."""
        errors: list[str] = []

        if self.tts_backend == "xtts":
            if not os.path.exists(self.xtts_voice_sample):
                errors.append(
                    f"Voice sample not found: '{self.xtts_voice_sample}'\n"
                    "    Record 10-30s of your voice:\n"
                    '    python -c "'
                    "import sounddevice as sd, scipy.io.wavfile as w, numpy as np; "
                    "a=sd.rec(int(20*22050),22050,1,'int16'); sd.wait(); "
                    "w.write('voice_sample.wav',22050,a)\""
                )
        elif self.tts_backend == "rvc":
            if not os.path.exists(self.rvc_model_path):
                errors.append(
                    f"RVC model not found: '{self.rvc_model_path}'\n"
                    "    Train via RVC WebUI and update rvc_model_path in config.py"
                )

        if errors:
            raise ValueError(
                "Atlas configuration errors:\n" + "\n".join(f"  • {e}" for e in errors)
            )
