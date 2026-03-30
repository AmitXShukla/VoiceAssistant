"""
Atlas — Complete Local Voice AI Assistant
==========================================
Wake word : "Hey Atlas"
STT       : faster-whisper  (CUDA)
LLM       : Ollama streaming + tool calling
TTS       : Coqui XTTS-v2 / RVC
Memory    : SQLite
UI        : FastAPI web dashboard

Usage:
    python main.py
    python main.py --tts rvc
    python main.py --no-ui
    python main.py --setup-wake-word
    python main.py --list-devices
"""

from __future__ import annotations

import argparse
import logging
import sys
import threading

# ── Logging setup (before any imports that log) ──────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("atlas.log", encoding="utf-8"),
    ],
)
log = logging.getLogger("atlas.main")


def main():
    parser = argparse.ArgumentParser(description="Atlas Voice Assistant")
    parser.add_argument("--no-ui", action="store_true", help="Disable web UI")
    parser.add_argument(
        "--tts",
        choices=["xtts", "rvc"],
        default="xtts",
        help="TTS backend (default: xtts)",
    )
    parser.add_argument(
        "--setup-wake-word",
        action="store_true",
        help="Train custom 'Hey Atlas' wake word model",
    )
    parser.add_argument(
        "--list-devices", action="store_true", help="Print audio devices and exit"
    )
    args = parser.parse_args()

    if args.list_devices:
        import sounddevice as sd

        print(sd.query_devices())
        sys.exit(0)

    if args.setup_wake_word:
        from core.wake_word import setup_wake_word

        setup_wake_word()
        sys.exit(0)

    from config import Config

    cfg = Config()
    cfg.tts_backend = args.tts

    print("""
    ╔══════════════════════════════════════════╗
    ║       ⚡  ATLAS Voice Assistant           ║
    ║  Say "Hey Atlas" to start talking        ║
    ║  Speak while Atlas talks to interrupt    ║
    ║  Ctrl+C to quit                          ║
    ╚══════════════════════════════════════════╝
    """)

    if not args.no_ui:
        from ui.server import start_ui_server

        ui_thread = threading.Thread(
            target=start_ui_server,
            args=(cfg,),
            daemon=True,
            name="ui-server",
        )
        ui_thread.start()
        log.info(f"🌐 Dashboard → http://localhost:{cfg.ui_port}")

    from core.state_machine import AtlasStateMachine

    atlas = AtlasStateMachine(cfg)
    atlas.run()


if __name__ == "__main__":
    main()
