# REVV — Voice-Driven Documentation for a Car Service Workshop

REVV turns the conversation between a mechanic and a customer into structured
paperwork. Audio from the browser is transcribed locally with Whisper, then a local
LLM (Gemma via Ollama) extracts the customer and vehicle details into a live profile.
The profile can be exported to Apple Notes or sent to a Google Form.

Built by Antonia Gherghe and Luca-Andrei Gherghe (2026).

## How it works
```
recorder.html  →  FastAPI (merge13.py)  →  Whisper (speech-to-text)
  (browser mic)                          →  Gemma via Ollama (JSON extraction)
                                         →  live profile → Apple Notes / Google Form
```
- Everything runs locally, with no cloud APIs for transcription or extraction
- Models are preloaded at startup so the first request is fast
- The LLM re-runs every few new lines with the full conversation as context, and
  only fills in fields it can ground in the transcript

## Files
| File | Purpose |
|---|---|
| `merge13.py` | Main server (latest version): sessions, live profile, exports |
| `recorder.html` | Browser recorder that streams audio to the server |
| `3endpoint.py` | Earlier 3-endpoint version using MLX Whisper (Apple Silicon) |
| `fastpi1.py` | First prototype: transcription endpoint only |
| `test_flux_llm.py` | Command-line test of the audio → text → JSON flow |

## API (`merge13.py`)
| Method | Endpoint | Description |
|---|---|---|
| POST | `/transcribe` | Audio file → text |
| POST | `/extract-info` | Text → structured JSON |
| POST | `/process-audio` | Full pipeline, updates the live profile |
| POST | `/reset-session` | Start a new customer |
| POST | `/finalize-session` | Final LLM pass at the end of a session |

## Run it
```bash
pip install -r requirements.txt
ollama pull gemma4:e4b
cp .env.example .env        # then add your Google Form ID (optional)
uvicorn merge13:app --host 127.0.0.1 --port 8000
```
Then open `recorder.html` in the browser.
