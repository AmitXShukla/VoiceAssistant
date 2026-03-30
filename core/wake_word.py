"""
core/wake_word.py — "Hey Atlas" wake word detection via OpenWakeWord

Python 3.12 fixes:
- threading.Thread | None type hint (valid in 3.10+)
- exec() replaced with proper function calls in setup_wake_word()
- Added explicit int16 dtype for sounddevice stream (OpenWakeWord requires int16)
- Removed bare except in _run
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Callable

import numpy as np
import sounddevice as sd

from config import Config

log = logging.getLogger("atlas.wake_word")


class WakeWordDetector:
    def __init__(self, cfg: Config, on_detected: Callable[[], None]):
        self.cfg = cfg
        self.on_detected = on_detected
        self._running = False
        self._thread: threading.Thread | None = None
        self._model = None
        self._model_key: str = ""
        self._load_model()

    def _load_model(self):
        try:
            from openwakeword.model import Model
        except ImportError:
            raise ImportError(
                "openwakeword not installed.\n"
                "  Run: pip install openwakeword\n"
                "       pip install onnxruntime-gpu  (for GPU inference)"
            )

        if os.path.exists(self.cfg.wake_word_model_path):
            log.info(f"Loading custom wake word: {self.cfg.wake_word_model_path}")
            self._model = Model(
                wakeword_models=[self.cfg.wake_word_model_path],
                inference_framework="onnx",
            )
            self._model_key = os.path.splitext(
                os.path.basename(self.cfg.wake_word_model_path)
            )[0]
            log.info("✅ Custom 'Hey Atlas' wake word model loaded.")
        else:
            log.warning(
                f"Custom model not found at '{self.cfg.wake_word_model_path}'.\n"
                f"  Falling back to built-in '{self.cfg.wake_word_fallback}'.\n"
                f"  Train a custom model: python main.py --setup-wake-word"
            )
            self._model = Model(
                wakeword_models=[self.cfg.wake_word_fallback],
                inference_framework="onnx",
            )
            self._model_key = self.cfg.wake_word_fallback.replace("-", "_")

    def start(self):
        self._running = True
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="wake-word"
        )
        self._thread.start()
        log.info(f"👂 Wake word active — say '{self.cfg.wake_word}'")

    def stop(self):
        self._running = False

    def _run(self):
        sr = 16000
        chunk_size = 1280   # 80 ms at 16 kHz — required by OpenWakeWord

        try:
            with sd.InputStream(
                samplerate=sr,
                channels=1,
                device=self.cfg.input_device,
                dtype="int16",             # OpenWakeWord requires int16
                blocksize=chunk_size,
            ) as stream:
                while self._running:
                    chunk, _ = stream.read(chunk_size)
                    audio = chunk.flatten().astype(np.int16)
                    prediction = self._model.predict(audio)
                    score = float(prediction.get(self._model_key, 0.0))

                    if score >= self.cfg.wake_word_threshold:
                        log.debug(f"Wake word score: {score:.3f}")
                        self._model.reset()    # prevent double-trigger
                        self.on_detected()
                        time.sleep(1.5)        # cooldown

        except sd.PortAudioError as e:
            log.error(f"Audio device error in wake word detector: {e}")
        except Exception as e:
            log.error(f"Wake word detector crashed: {e}", exc_info=True)


def setup_wake_word():
    """
    Interactive helper to train a custom 'hey_atlas' wake word model.
    Requires: pip install openwakeword[training]
    """
    print("""
╔══════════════════════════════════════════════════════════╗
║        Hey Atlas — Custom Wake Word Training             ║
╚══════════════════════════════════════════════════════════╝

This will train a custom 'hey_atlas' ONNX model.
Estimated time: ~5 minutes on GPU.

Prerequisites:
    pip install openwakeword[training]
    """)

    try:
        import openwakeword  # noqa: F401
    except ImportError:
        print("❌ openwakeword[training] not installed.")
        print("   Run: pip install openwakeword[training]")
        return

    os.makedirs("wake_word/training_data", exist_ok=True)

    print("Step 1/3: Generating synthetic training data for 'hey atlas'…")
    try:
        from openwakeword.training import generate_data
        generate_data(
            positive_phrases=["hey atlas"],
            negative_phrases=["hey there", "hello atlas", "hey alice", "hey alexis",
                               "hey android", "hey at last"],
            output_dir="wake_word/training_data",
            num_positives=5000,
            num_negatives=10000,
        )
        print("  ✅ Training data generated.")
    except Exception as e:
        print(f"  ❌ Data generation failed: {e}")
        return

    print("Step 2/3: Training model…")
    try:
        from openwakeword.training import train_model
        train_model(
            training_data_dir="wake_word/training_data",
            output_dir="wake_word",
            model_name="hey_atlas",
            epochs=30,
            batch_size=256,
        )
        print("  ✅ Model trained.")
    except Exception as e:
        print(f"  ❌ Training failed: {e}")
        return

    print("""
Step 3/3: Done!
  Model saved to: wake_word/hey_atlas.onnx
  Restart Atlas to use the custom wake word.
    """)
