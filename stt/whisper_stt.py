"""
stt/whisper_stt.py — Speech-to-Text using faster-whisper (CUDA)

Python 3.12 fixes:
- Added explicit VAD parameters dict to avoid positional-arg deprecation
- Model pre-loaded in __init__ to avoid race condition on first request
- Added decode error handling for unusual audio
- condition_on_previous_text only set when there are previous segments
"""

from __future__ import annotations

import logging
import os

from config import Config

log = logging.getLogger("atlas.stt")

_model = None


# def _get_model(cfg: Config):
#     global _model
#     if _model is None:
#         from faster_whisper import WhisperModel
#         log.info(
#             f"Loading Whisper '{cfg.whisper_model}' "
#             f"on {cfg.whisper_device} [{cfg.whisper_compute_type}]…"
#         )
#         _model = WhisperModel(
#             cfg.whisper_model,
#             device=cfg.whisper_device,
#             compute_type=cfg.whisper_compute_type,
#         )
#         log.info("✅ Whisper loaded.")
#     return _model


def _get_model(cfg: Config):
    global _model
    if _model is None:
        import whisper

        log.info(f"Loading Whisper '{cfg.whisper_model}' on {cfg.whisper_device}…")
        _model = whisper.load_model(cfg.whisper_model, device=cfg.whisper_device)
        log.info("✅ Whisper loaded.")
    return _model


class WhisperSTT:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        _get_model(cfg)  # Pre-load; warms GPU memory before first request

    # def transcribe(self, audio_path: str) -> str:
    #     model = _get_model(self.cfg)

    #     segments, info = model.transcribe(
    #         audio_path,
    #         language=self.cfg.whisper_language,
    #         beam_size=5,
    #         vad_filter=True,
    #         vad_parameters={
    #             "min_silence_duration_ms": 300,
    #             "speech_pad_ms": 200,
    #         },
    #         condition_on_previous_text=False,  # avoids hallucination loops
    #         no_speech_threshold=0.6,
    #         log_prob_threshold=-1.0,
    #     )

    #     # Collect all segments — segments is a generator, consume it fully
    #     text_parts = []
    #     for seg in segments:
    #         cleaned = seg.text.strip()
    #         if cleaned:
    #             text_parts.append(cleaned)

    #     text = " ".join(text_parts)
    #     log.debug(
    #         f"STT: {text!r} "
    #         f"(lang={info.language}, prob={info.language_probability:.2f})"
    #     )

    #     try:
    #         os.remove(audio_path)
    #     except OSError:
    #         pass

    #     return text
    def transcribe(self, audio_path: str) -> str:
        model = _get_model(self.cfg)
        result = model.transcribe(
            audio_path,
            language=self.cfg.whisper_language,
            fp16=(self.cfg.whisper_device == "cuda"),
        )
        text = result["text"].strip()
        log.debug(f"STT: {text!r}")
        try:
            os.remove(audio_path)
        except OSError:
            pass
        return text
