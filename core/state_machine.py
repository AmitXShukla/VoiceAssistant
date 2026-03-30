"""
core/state_machine.py — Atlas State Machine

Python 3.12 fixes:
- Fixed THINKING→SPEAKING logic: state now transitions immediately when
  LLM+TTS threads start, so TTS_DONE fires correctly from SPEAKING state
- Lazy import of broadcast_state to avoid circular-import at module load
- Optional[X] → X | None style throughout
- threading.Thread | None type hints (valid in 3.10+)
"""

from __future__ import annotations

import logging
import queue
import threading
from enum import Enum, auto

from config import Config
from core.audio_pipeline import AudioPipeline
from core.wake_word import WakeWordDetector
from stt.whisper_stt import WhisperSTT
from llm.ollama_client import OllamaClient
from memory.store import MemoryStore

log = logging.getLogger("atlas.state_machine")


class State(Enum):
    IDLE = auto()
    LISTENING = auto()
    TRANSCRIBING = auto()
    THINKING = auto()
    SPEAKING = auto()
    ERROR = auto()


class Event(Enum):
    WAKE_WORD_DETECTED = auto()
    SPEECH_ENDED = auto()
    TRANSCRIPTION_DONE = auto()
    TTS_DONE = auto()
    INTERRUPT = auto()
    ERROR = auto()
    RESET = auto()


def _broadcast_state(state: str, text: str | None = None) -> None:
    """Lazy import wrapper — avoids circular import at module load time."""
    try:
        from ui.server import broadcast_state
        broadcast_state(state, text)
    except Exception:
        pass  # UI may not be running


class AtlasStateMachine:
    def __init__(self, cfg: Config):
        self.cfg = cfg
        self.state = State.IDLE
        self._event_queue: queue.Queue = queue.Queue()
        self._stop_event = threading.Event()
        self._tts_stop_event = threading.Event()

        log.info("⚙️  Initialising components…")
        cfg.validate()

        self.memory = MemoryStore(cfg)
        self.audio = AudioPipeline(cfg)
        self.wake_detector = WakeWordDetector(cfg, self._on_wake_word)
        self.stt = WhisperSTT(cfg)
        self.llm = OllamaClient(cfg, self.memory)

        # Import TTS backend lazily based on config
        if cfg.tts_backend == "xtts":
            from tts.xtts_backend import XTTSBackend
            self.tts = XTTSBackend(cfg)
        else:
            from tts.rvc_backend import RVCBackend
            self.tts = RVCBackend(cfg)

        log.info("✅ All components initialised.")

    # ── Public ────────────────────────────────────────────────────────────────

    def run(self):
        log.info("🟢 Atlas is running. Say 'Hey Atlas' to begin.")
        self.wake_detector.start()
        self.audio.start()

        try:
            while not self._stop_event.is_set():
                try:
                    event, data = self._event_queue.get(timeout=0.1)
                    self._handle_event(event, data)
                except queue.Empty:
                    continue
        except KeyboardInterrupt:
            log.info("Shutting down…")
        finally:
            self._shutdown()

    def emit(self, event: Event, data=None):
        self._event_queue.put((event, data))

    # ── State transitions ─────────────────────────────────────────────────────

    def _handle_event(self, event: Event, data):
        log.debug(f"  Event={event.name}  State={self.state.name}")

        # Global events handled regardless of state
        if event == Event.INTERRUPT:
            self._handle_interrupt()
            return

        if event in (Event.RESET, Event.ERROR):
            if event == Event.ERROR:
                log.error(f"Error: {data}")
                self._speak_error()
            self._transition(State.IDLE)
            return

        # State-specific transitions
        if self.state == State.IDLE:
            if event == Event.WAKE_WORD_DETECTED:
                self._transition(State.LISTENING)
                self._start_listening()

        elif self.state == State.LISTENING:
            if event == Event.SPEECH_ENDED:
                self._transition(State.TRANSCRIBING)
                self._start_transcription(data)

        elif self.state == State.TRANSCRIBING:
            if event == Event.TRANSCRIPTION_DONE:
                if not data or not data.strip():
                    log.info("Empty transcription — returning to IDLE.")
                    self._transition(State.IDLE)
                    return
                self._transition(State.THINKING)
                self._start_llm(data)

        elif self.state == State.SPEAKING:
            if event == Event.TTS_DONE:
                self.audio.stop_interrupt_detection()
                self._transition(State.IDLE)

    def _transition(self, new_state: State):
        log.info(f"  {self.state.name} → {new_state.name}")
        self.state = new_state
        _broadcast_state(new_state.name)

    # ── Action methods ────────────────────────────────────────────────────────

    def _on_wake_word(self):
        log.info("🎯 Wake word detected!")
        self.emit(Event.WAKE_WORD_DETECTED)

    def _start_listening(self):
        self.audio.play_chime("start")
        self.audio.start_recording(
            on_speech_end=lambda path: self.emit(Event.SPEECH_ENDED, path)
        )
        log.info("🎤 Listening…")

    def _start_transcription(self, audio_path: str):
        log.info("📝 Transcribing…")

        def _run():
            try:
                text = self.stt.transcribe(audio_path)
                log.info(f"  You: {text!r}")
                _broadcast_state("TRANSCRIBING", text)
                self.emit(Event.TRANSCRIPTION_DONE, text)
            except Exception as exc:
                self.emit(Event.ERROR, str(exc))

        threading.Thread(target=_run, daemon=True, name="stt").start()

    def _start_llm(self, user_text: str):
        """
        Start LLM + streaming TTS pipeline.

        Fix vs original: we transition to SPEAKING immediately here so that
        TTS_DONE (emitted by the TTS consumer) is processed in the correct
        SPEAKING state. Previously the state stayed THINKING and TTS_DONE
        was silently dropped.
        """
        log.info("🤖 Thinking…")
        self._tts_stop_event.clear()

        # Sentence queue — LLM thread produces, TTS consumer thread consumes
        sentence_queue: queue.Queue = queue.Queue()

        # Transition to SPEAKING now so TTS_DONE is handled correctly
        self._transition(State.SPEAKING)
        self.audio.start_interrupt_detection(
            callback=lambda: self.emit(Event.INTERRUPT)
        )

        # TTS consumer thread
        threading.Thread(
            target=self._tts_consumer,
            args=(sentence_queue,),
            daemon=True,
            name="tts-consumer",
        ).start()

        # LLM streaming thread
        def _run_llm():
            try:
                full_response = self.llm.stream_response(
                    user_text,
                    sentence_callback=lambda s: sentence_queue.put(s),
                    stop_event=self._tts_stop_event,
                )
                sentence_queue.put(None)   # sentinel: LLM done
                _broadcast_state("SPEAKING", full_response)
            except Exception as exc:
                log.error(f"LLM error: {exc}", exc_info=True)
                sentence_queue.put(None)
                self.emit(Event.ERROR, str(exc))

        threading.Thread(target=_run_llm, daemon=True, name="llm").start()

    def _tts_consumer(self, sentence_queue: queue.Queue):
        """
        Pulls sentences from the queue, synthesizes each, and plays immediately.
        Checks _tts_stop_event to support interrupts.
        """
        while True:
            if self._tts_stop_event.is_set():
                break
            try:
                sentence = sentence_queue.get(timeout=0.1)
            except queue.Empty:
                continue

            if sentence is None:
                break  # LLM finished

            if not sentence.strip():
                continue

            try:
                audio, sr = self.tts.synthesize(sentence)
                if not self._tts_stop_event.is_set():
                    self.audio.play(audio, sr, stop_event=self._tts_stop_event)
            except Exception as exc:
                log.error(f"TTS error on sentence '{sentence[:40]}': {exc}")

        if not self._tts_stop_event.is_set():
            self.emit(Event.TTS_DONE)

    def _handle_interrupt(self):
        if self.state not in (State.SPEAKING, State.THINKING):
            return
        log.info("⚡ Interrupt!")
        self._tts_stop_event.set()
        self.audio.stop_interrupt_detection()
        self.audio.play_chime("interrupt")
        self._transition(State.LISTENING)
        self._start_listening()

    def _speak_error(self):
        try:
            audio, sr = self.tts.synthesize("Sorry, something went wrong. Please try again.")
            self.audio.play(audio, sr)
        except Exception:
            pass

    def _shutdown(self):
        log.info("Shutting down Atlas…")
        self._stop_event.set()
        self._tts_stop_event.set()
        self.wake_detector.stop()
        self.audio.stop()
        self.memory.close()
        log.info("👋 Goodbye.")
