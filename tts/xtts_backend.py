"""
tts/xtts_backend.py — Coqui XTTS-v2 voice cloning TTS

Python 3.12 fixes:
- Added torch.no_grad() context manager (required in newer torch/TTS versions)
- Explicit numpy float32 conversion with np.asarray instead of np.array
- Handle both old TTS API (synthesizer.tts_model) and newer direct API
- Module-level singletons (_model, _latents) shared safely across instances
- Better error message if voice sample is too short
"""

from __future__ import annotations

import logging
import os

import numpy as np
import torch

from config import Config

log = logging.getLogger("atlas.tts.xtts")

_model = None
_latents: tuple | None = None


def _init_model(cfg: Config):
    global _model
    if _model is not None:
        return _model
    try:
        from TTS.api import TTS
    except ImportError:
        raise ImportError("Coqui TTS not installed.\n  Run: pip install TTS")
    log.info(f"Loading XTTS-v2 on {cfg.xtts_device}…")
    tts = TTS(cfg.xtts_model_name)
    tts.to(cfg.xtts_device)
    _model = tts
    log.info("✅ XTTS-v2 loaded.")
    return _model


def _init_latents(cfg: Config, model):
    global _latents
    if _latents is not None:
        return _latents

    if not os.path.exists(cfg.xtts_voice_sample):
        raise FileNotFoundError(
            f"Voice sample not found: '{cfg.xtts_voice_sample}'\n"
            "  Record 10-30 seconds of your voice and save as voice_sample.wav\n"
            '  Quick record: python -c "'
            "import sounddevice as sd, scipy.io.wavfile as w, numpy as np; "
            "a=sd.rec(int(20*22050),22050,1,'int16'); sd.wait(); "
            "w.write('voice_sample.wav',22050,a)\""
        )

    log.info(f"Computing speaker latents from: {cfg.xtts_voice_sample}")
    try:
        # Try newer TTS API path
        gpt_latent, spk_embed = model.synthesizer.tts_model.get_conditioning_latents(
            audio_path=[cfg.xtts_voice_sample]
        )
    except AttributeError:
        # Fall back for different TTS versions
        try:
            gpt_latent, spk_embed = model.tts_model.get_conditioning_latents(
                audio_path=[cfg.xtts_voice_sample]
            )
        except AttributeError as e:
            raise RuntimeError(
                f"Could not access XTTS model internals: {e}\n"
                "Try updating TTS: pip install --upgrade TTS"
            )

    _latents = (gpt_latent, spk_embed)
    log.info("Speaker latents cached — voice clone ready.")
    return _latents


class XTTSBackend:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._model = _init_model(cfg)
        self._latents = _init_latents(cfg, self._model)
        # Warmup
        log.info("Warming up TTS…")
        self.synthesize("Warming up.")
        log.info("✅ TTS warm.")

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        Synthesize text in your cloned voice.
        Returns (audio_float32, sample_rate).
        """
        gpt_latent, spk_embed = self._latents

        try:
            with torch.no_grad():
                outputs = self._model.synthesizer.tts_model.inference(
                    text=text,
                    language=self.cfg.xtts_language,
                    gpt_cond_latent=gpt_latent,
                    speaker_embedding=spk_embed,
                    temperature=0.65,
                    length_penalty=1.0,
                    repetition_penalty=5.0,
                    top_k=50,
                    top_p=0.85,
                )
        except AttributeError:
            # Newer TTS API fallback
            with torch.no_grad():
                outputs = self._model.tts_model.inference(
                    text=text,
                    language=self.cfg.xtts_language,
                    gpt_cond_latent=gpt_latent,
                    speaker_embedding=spk_embed,
                    temperature=0.65,
                    repetition_penalty=5.0,
                    top_k=50,
                    top_p=0.85,
                )

        audio = outputs["wav"]
        sr = self._model.synthesizer.output_sample_rate

        # Ensure numpy float32
        audio = np.asarray(audio, dtype=np.float32)

        return audio, int(sr)
