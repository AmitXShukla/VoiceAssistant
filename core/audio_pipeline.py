"""
core/audio_pipeline.py — Mic input, Silero VAD, noise suppression, playback

Python 3.12 fixes:
- Silero VAD v5 API: handle both old tuple-based and new direct import APIs
- Removed unused speech_frames variable
- Guard DeepFilterNet attributes with hasattr before accessing
- Optional[X] → X | None (PEP 604 style, valid in 3.10+)
- Explicit numpy float32 cast before torch.from_numpy (avoids dtype warnings)
"""

from __future__ import annotations

import logging
import os
import tempfile
import threading
import time
from typing import Callable

import numpy as np
import scipy.io.wavfile as wavfile
import scipy.signal as signal
import sounddevice as sd
import torch

from config import Config

log = logging.getLogger("atlas.audio")


def _generate_chime(
    freq: float = 880, duration: float = 0.12, sr: int = 22050, fade: float = 0.01
) -> np.ndarray:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    wave = 0.4 * np.sin(2 * np.pi * freq * t)
    fade_n = int(sr * fade)
    wave[:fade_n] *= np.linspace(0, 1, fade_n)
    wave[-fade_n:] *= np.linspace(1, 0, fade_n)
    return wave.astype(np.float32)


CHIMES = {
    "start": _generate_chime(880, 0.12),
    "interrupt": _generate_chime(440, 0.10),
    "error": _generate_chime(220, 0.20),
}


class AudioPipeline:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self._running = False
        self._capture_thread: threading.Thread | None = None
        self._interrupt_thread: threading.Thread | None = None
        self._interrupt_stop = threading.Event()
        self._interrupt_callback: Callable | None = None
        self._vad_model = None
        self._load_vad()
        if cfg.deepfilter:
            self._load_deepfilter()

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(self):
        self._running = True
        log.info("🎙 Audio pipeline started.")

    def stop(self):
        self._running = False
        self._interrupt_stop.set()

    # ── VAD ───────────────────────────────────────────────────────────────────

    def _load_vad(self):
        """
        Load Silero VAD. Handles both the v4 (torch.hub tuple return) and
        v5 (silero_vad package direct import) APIs.
        """
        log.info("Loading Silero VAD…")

        # Try the newer silero-vad package first (pip install silero-vad)
        try:
            from silero_vad import load_silero_vad

            self._vad_model = load_silero_vad()
            log.info("Silero VAD loaded (silero-vad package).")
            return
        except ImportError:
            pass

        # Fall back to torch.hub (original method)
        try:
            model, _ = torch.hub.load(
                repo_or_dir="snakers4/silero-vad",
                model="silero_vad",
                force_reload=False,
                trust_repo=True,
                verbose=False,
            )
            self._vad_model = model
            log.info("Silero VAD loaded (torch.hub).")
        except Exception as e:
            log.error(f"Failed to load Silero VAD: {e}")
            raise

    def _load_deepfilter(self):
        try:
            from df.enhance import enhance, init_df

            self._df_model, self._df_state, _ = init_df()
            self._df_enhance = enhance
            log.info("DeepFilterNet loaded.")
        except ImportError:
            log.warning("DeepFilterNet not installed — falling back to noisereduce.")
            self.cfg.deepfilter = False

    # ── Recording ─────────────────────────────────────────────────────────────

    def start_recording(self, on_speech_end: Callable[[str], None]):
        t = threading.Thread(
            target=self._record_with_vad,
            args=(on_speech_end,),
            daemon=True,
            name="record",
        )
        t.start()

    def _record_with_vad(self, on_speech_end: Callable[[str], None]):
        cfg = self.cfg
        sr = cfg.samplerate
        # chunk_ms = 96          # Silero VAD optimal chunk size at 16 kHz
        # chunk_size = int(sr * chunk_ms / 1000)
        chunk_size = 512

        frames: list[np.ndarray] = []
        in_speech = False
        silent_chunks = 0
        # max_silent = int(cfg.vad_min_silence_ms / chunk_ms)
        # max_chunks = int(cfg.max_record_seconds * 1000 / chunk_ms)
        max_silent = int(cfg.vad_min_silence_ms / (chunk_size / sr * 1000))
        max_chunks = int(cfg.max_record_seconds * sr / chunk_size)

        try:
            with sd.InputStream(
                samplerate=sr,
                channels=1,
                device=cfg.input_device,
                dtype="float32",
                blocksize=chunk_size,
            ) as stream:
                for _ in range(max_chunks):
                    chunk, _ = stream.read(chunk_size)
                    # Ensure contiguous float32 array before creating tensor
                    audio_chunk = np.ascontiguousarray(
                        chunk.flatten(), dtype=np.float32
                    )
                    tensor = torch.from_numpy(audio_chunk)

                    # Silero VAD inference — returns scalar probability
                    with torch.no_grad():
                        speech_prob = float(self._vad_model(tensor, sr).item())

                    if speech_prob > cfg.vad_threshold:
                        in_speech = True
                        silent_chunks = 0
                        frames.append(audio_chunk)
                    elif in_speech:
                        frames.append(audio_chunk)
                        silent_chunks += 1
                        if silent_chunks >= max_silent:
                            log.debug("End of speech detected.")
                            break
                    # else: pre-speech silence — skip

        except Exception as e:
            log.error(f"Recording error: {e}", exc_info=True)
            return

        if not frames:
            log.info("No speech detected in recording.")
            return

        audio = np.concatenate(frames)
        audio = self._denoise(audio, sr)

        audio_int16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
        wavfile.write(tmp.name, sr, audio_int16)

        if cfg.log_audio:
            os.makedirs(cfg.audio_log_dir, exist_ok=True)
            wavfile.write(
                f"{cfg.audio_log_dir}/{int(time.time())}.wav", sr, audio_int16
            )

        on_speech_end(tmp.name)

    def _denoise(self, audio: np.ndarray, sr: int) -> np.ndarray:
        if self.cfg.deepfilter and hasattr(self, "_df_model"):
            try:
                tensor = torch.from_numpy(audio).unsqueeze(0)
                enhanced = self._df_enhance(self._df_model, self._df_state, tensor)
                return enhanced.squeeze(0).numpy()
            except Exception as e:
                log.warning(f"DeepFilterNet denoising failed: {e}")

        if self.cfg.noise_suppress:
            try:
                import noisereduce as nr

                reduced = nr.reduce_noise(y=audio, sr=sr, stationary=False)
                return reduced.astype(np.float32)
            except ImportError:
                log.debug("noisereduce not installed — skipping noise suppression.")
            except Exception as e:
                log.warning(f"noisereduce failed: {e}")

        return audio

    # ── Playback ──────────────────────────────────────────────────────────────

    def play(
        self, audio: np.ndarray, sr: int, stop_event: threading.Event | None = None
    ):
        """Play audio synchronously. Stops early if stop_event is set."""
        if audio.dtype != np.float32:
            audio = audio.astype(np.float32) / 32768.0

        if sr != self.cfg.samplerate:
            num_samples = int(len(audio) * self.cfg.samplerate / sr)
            audio = signal.resample(audio, num_samples).astype(np.float32)
            sr = self.cfg.samplerate

        chunk_size = int(sr * 0.05)  # 50 ms chunks — allows stop_event polling
        with sd.OutputStream(
            samplerate=sr,
            channels=1,
            device=self.cfg.output_device,
            dtype="float32",
        ) as stream:
            for i in range(0, len(audio), chunk_size):
                if stop_event and stop_event.is_set():
                    break
                stream.write(audio[i : i + chunk_size])

    def play_chime(self, name: str):
        chime = CHIMES.get(name, CHIMES["start"])
        sd.play(chime, samplerate=22050, device=self.cfg.output_device)
        sd.wait()

    # ── Interrupt detection ───────────────────────────────────────────────────

    def start_interrupt_detection(self, callback: Callable):
        self._interrupt_stop.clear()
        self._interrupt_callback = callback
        threading.Thread(
            target=self._detect_interrupt,
            daemon=True,
            name="interrupt",
        ).start()

    def stop_interrupt_detection(self):
        self._interrupt_stop.set()

    def _detect_interrupt(self):
        cfg = self.cfg
        sr = cfg.samplerate
        chunk_size = int(sr * 0.05)
        hold_chunks = max(1, int(cfg.interrupt_hold_ms / 50))
        triggered = 0

        try:
            with sd.InputStream(
                samplerate=sr,
                channels=1,
                device=cfg.input_device,
                dtype="float32",
                blocksize=chunk_size,
            ) as stream:
                while not self._interrupt_stop.is_set():
                    chunk, _ = stream.read(chunk_size)
                    rms = float(np.sqrt(np.mean(chunk**2)))
                    if rms > cfg.interrupt_rms_threshold:
                        triggered += 1
                        if triggered >= hold_chunks:
                            log.debug("Interrupt triggered.")
                            self._interrupt_stop.set()
                            if self._interrupt_callback:
                                self._interrupt_callback()
                            return
                    else:
                        triggered = max(0, triggered - 1)
        except Exception as e:
            log.debug(f"Interrupt detector exited: {e}")
