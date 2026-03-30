# ⚡ Jarvis — Complete Local Voice AI Assistant

A fully offline voice assistant that runs entirely on your NVIDIA GPU.
No cloud, no API keys, no subscriptions.

```
Say "Hey Jarvis"  →  speak your request  →  Jarvis responds in your voice
```

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     AUDIO INPUT THREAD                           │
│   Mic → noisereduce/DeepFilterNet → Silero VAD → Ring Buffer    │
└──────────────────────┬──────────────────────────────────────────┘
                       │
┌──────────────────────▼──────────────────────────────────────────┐
│                  STATE MACHINE (main thread)                      │
│                                                                   │
│  IDLE ──[Hey Atlas]──► LISTENING ──[silence]──► TRANSCRIBING     │
│    ▲                                                  │           │
│    │                                                  ▼           │
│  TTS_DONE ◄── SPEAKING ◄── THINKING ◄────────── text ready       │
│                  │                                                │
│           [interrupt]──────────────────────► LISTENING           │
└──────────┬────────────────────────────────────────────────────--─┘
           │
┌──────────▼──────────────┐  ┌────────────────────────────────────┐
│     LLM (Ollama)        │  │   Streaming TTS (XTTS-v2)          │
│  gpt:oss:120b           │  │  sentence queue → synth → play     │
│  streaming + tool calls │→─│  (speaks while LLM still typing)   │
└─────────────────────────┘  └────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                        TOOL REGISTRY                              │
│     web_search │ get_datetime │ set_reminder │ list_reminders    │
└──────────┬──────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│                  MEMORY (SQLite)                                  │
│        messages │ reminders │ user_facts                         │
└─────────────────────────────────────────────────────────────────┘
           │
┌──────────▼──────────────────────────────────────────────────────┐
│             WEB DASHBOARD  http://localhost:7860                  │
│   Live state orb │ Conversation history │ Reminders              │
└─────────────────────────────────────────────────────────────────┘
```

---

## Quick Start

### 1. Prerequisites

- NVIDIA GPU (RTX 3080+ recommended)
- CUDA 12.1+ with cuDNN
- Python 3.10+
- [Ollama](https://ollama.com) installed and running

```bash
# Pull your model
ollama pull llama3:70b       # or your gpt:oss:120b tag
ollama serve
```

### 2. Install

```bash
# Clone / download this folder, then:
cd atlas

# PyTorch FIRST (match your CUDA version)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121

# Everything else
pip install -r requirements.txt
```

### 3. Record your voice sample

```bash
python -c "
import sounddevice as sd, scipy.io.wavfile as wav, numpy as np
print('Recording 20 seconds of your voice...')
print('Speak naturally — tell a story, read a paragraph, anything.')
audio = sd.rec(int(20 * 22050), samplerate=22050, channels=1, dtype='int16')
sd.wait()
wav.write('voice_sample.wav', 22050, audio)
print('Saved: voice_sample.wav')
"
```

**Tips for best cloning quality:**
- Quiet room, no echo
- Normal speaking pace and volume
- Varied sentences (not a word list)
- 10-30 seconds is ideal

### 4. (Optional) Train custom "Hey Atlas" wake word

```bash
pip install openwakeword[training]
python main.py --setup-wake-word
# ~5 minutes, saves to wake_word/hey_atlas.onnx
```

During development, Atlas uses the built-in `hey_jarvis` model as a fallback.

### 5. Configure

Edit `config.py` — key settings:

```python
ollama_model = "gpt:oss:120b"        # your Ollama model tag
xtts_voice_sample = "voice_sample.wav"
whisper_model = "large-v3"           # or "medium" for less VRAM
```

### 6. Run!

```bash
python main.py
```

Open **http://localhost:7860** to see the live dashboard.

---

## Usage

| Say | What happens |
|---|---|
| `"Hey Atlas"` | Wakes up, starts listening |
| `"Hey Atlas, what time is it?"` | Uses get_datetime tool |
| `"Hey Atlas, search for…"` | Uses DuckDuckGo web search |
| `"Hey Atlas, remind me to drink water in 10 minutes"` | Sets a 600-second reminder |
| `"Hey Atlas, what reminders do I have?"` | Lists active reminders |
| Speak while Atlas is talking | Interrupts and starts listening |

---

## Command Line Options

```bash
python main.py                    # default (XTTS-v2 + web UI)
python main.py --tts rvc          # use RVC backend
python main.py --no-ui            # terminal only, no web dashboard
python main.py --setup-wake-word  # train custom wake word model
python main.py --list-devices     # list audio input/output devices
```

---

## VRAM Requirements

| Component | VRAM |
|---|---|
| faster-whisper large-v3 | ~1.5 GB |
| XTTS-v2 | ~3 GB |
| Silero VAD | ~0.1 GB |
| Ollama 120B Q4 | ~70 GB |
| **Total** | **~75 GB** |

For a single GPU setup, use a smaller model (`llama3:70b` = ~40 GB Q4).
For the full 120B you'll want an A100/H100 or multi-GPU with NVLink.

### Reduce VRAM usage:
```python
# In config.py:
whisper_model = "medium"           # saves ~1 GB
whisper_compute_type = "int8"      # halves Whisper VRAM
```

---

## Project Structure

```
atlas/
├── main.py                 Entry point
├── config.py               All configuration
├── requirements.txt
├── voice_sample.wav        ← You provide this
├── wake_word/
│   └── hey_atlas.onnx      ← Generated by --setup-wake-word
├── core/
│   ├── state_machine.py    FSM orchestrator
│   ├── audio_pipeline.py   Mic input, VAD, noise suppression, playback
│   └── wake_word.py        OpenWakeWord detector
├── stt/
│   └── whisper_stt.py      faster-whisper transcription
├── llm/
│   ├── ollama_client.py    Streaming Ollama client
│   └── tool_registry.py    Tool call dispatcher
├── tts/
│   ├── xtts_backend.py     XTTS-v2 voice cloning
│   └── rvc_backend.py      RVC pipeline
├── tools/
│   ├── web_search.py       DuckDuckGo search
│   ├── datetime_tool.py    Date & time
│   └── reminders.py        Timers & reminders
├── memory/
│   └── store.py            SQLite conversation memory
└── ui/
    ├── server.py            FastAPI + WebSocket server
    └── static/
        └── index.html       Web dashboard
```

---

## Troubleshooting

**"Cannot connect to Ollama"**
→ Run `ollama serve` in a separate terminal

**"Voice sample not found"**
→ Record `voice_sample.wav` (see Step 3 above)

**Wake word not triggering**
→ Lower `wake_word_threshold` in config.py (try 0.5)
→ Train the custom model: `python main.py --setup-wake-word`

**Too many false wake word triggers**
→ Raise `wake_word_threshold` (try 0.85)

**High latency on first response**
→ Normal — models are loading. Second response will be fast.

**Audio device issues**
→ Run `python main.py --list-devices` and set `input_device` / `output_device` in config.py

**CUDA out of memory**
→ Set `whisper_model = "medium"` and `whisper_compute_type = "int8"` in config.py
