"""
tts/rvc_backend.py — RVC voice conversion TTS

Python 3.12 fix:
- Replace asyncio.run() with asyncio.new_event_loop() to avoid
  "There is no current event loop" errors in Python 3.12 threads.
- asyncio.run() can fail if the calling thread was created by an
  async framework. new_event_loop() is always safe in a plain thread.
"""

from __future__ import annotations

import asyncio
import logging
import os
import tempfile

import numpy as np
import scipy.io.wavfile as wav_io

from config import Config

log = logging.getLogger("atlas.tts.rvc")

_rvc = None


def _run_async(coro):
    """Run a coroutine safely from any thread (Python 3.12 compatible)."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class RVCBackend:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._rvc = None
        self._init()

    def _init(self):
        global _rvc
        if _rvc is None:
            try:
                from rvc_python.infer import RVCInference
            except ImportError:
                raise ImportError(
                    "rvc-python not installed.\n"
                    "  Run: pip install rvc-python\n"
                    "  Or use --tts xtts (recommended)"
                )
            if not os.path.exists(self.cfg.rvc_model_path):
                raise FileNotFoundError(
                    f"RVC model not found: '{self.cfg.rvc_model_path}'\n"
                    "  Train a model via RVC WebUI and set rvc_model_path in config.py"
                )
            log.info(f"Loading RVC from {self.cfg.rvc_model_path}…")
            _rvc = RVCInference(device=self.cfg.rvc_device)
            _rvc.load_model(self.cfg.rvc_model_path)
            log.info("RVC loaded. Warming up…")
            self._rvc = _rvc
            self.synthesize("Ready.")
            log.info("✅ RVC warm.")
        self._rvc = _rvc

    async def _edge_tts_save(self, text: str, path: str):
        import edge_tts
        await edge_tts.Communicate(text, self.cfg.rvc_base_tts_voice).save(path)

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """
        edge-tts → RVC voice conversion.
        Returns (audio_float32, sample_rate).
        """
        base_tmp = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
        out_tmp  = tempfile.NamedTemporaryFile(suffix=".wav",  delete=False)
        base_tmp.close()
        out_tmp.close()

        try:
            # Python 3.12 safe async execution
            _run_async(self._edge_tts_save(text, base_tmp.name))

            self._rvc.infer(
                input_path=base_tmp.name,
                output_path=out_tmp.name,
                f0_up_key=self.cfg.rvc_pitch_shift,
                f0_method=self.cfg.rvc_f0_method,
                index_path=self.cfg.rvc_index_path or None,
                index_rate=0.75,
                protect=0.33,
            )

            sr, audio = wav_io.read(out_tmp.name)
            if audio.dtype == np.int16:
                audio = audio.astype(np.float32) / 32768.0
            elif audio.dtype == np.int32:
                audio = audio.astype(np.float32) / 2147483648.0

            return audio.astype(np.float32), int(sr)

        finally:
            for f in (base_tmp.name, out_tmp.name):
                try:
                    os.remove(f)
                except OSError:
                    pass
