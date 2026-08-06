import os
import shutil
import tempfile
import json
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
import whisper
import ollama

app = FastAPI(title="Audio Transcription API")

# Global variable to hold the loaded model instance
model = None
device = None

# ---- AICI SCHIMBI CE MODEL LLM TESTEZI ----
LLM_MODEL = "gemma4:e4b"

EXTENSII_ACCEPTATE = (".mp3", ".wav")


@app.on_event("startup")
def load_model():
    """Load the Whisper model once when the FastAPI server starts."""
    global model, device
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Se încarcă modelul Whisper în memorie (device: {device})...")
    # Loading the model once globally saves time on every request
    model = whisper.load_model("turbo", device=device)
    print("Modelul Whisper a fost încărcat cu succes!")

    print(f"Se încarcă modelul LLM ({LLM_MODEL}) în memorie prin Ollama...")
    # Un apel "gol" fortam Ollama sa incarce modelul in memorie acum,
    # nu la prima cerere reala - la fel ca la Whisper mai sus
    ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": ""}])
    print("Modelul LLM a fost încărcat cu succes!")

def extrage_json(text: str) -> str:
    """Trimite textul transcris catre LLM si cere extragere structurata in JSON."""
    prompt = f"""Extrage din textul urmator informatii despre o programare la service auto.
Raspunde STRICT cu un obiect JSON, fara alt text, fara ```json.
Daca o informatie nu apare in text, pune valoarea "-" (liniuta), nu inventa date.

{{"marca": "", "model": "", "an": null, "culoare": "", "motiv_vizita": "", "kilometri": "", "vin": "", "descriere_extra": ""}}

Text: "{text}"
"""
    raspuns = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return raspuns["message"]["content"]


def completeaza_valori_lipsa(date: dict) -> dict:
    """Inlocuieste orice valoare goala/lipsa cu '-'."""
    for cheie, valoare in date.items():
        if valoare is None or (isinstance(valoare, str) and valoare.strip() == ""):
            date[cheie] = "-"
    return date


@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # 1. Validate file extension
    if not file.filename.lower().endswith(EXTENSII_ACCEPTATE):
        raise HTTPException(
            status_code=400,
            detail=f"Format neacceptat. Formate acceptate: {EXTENSII_ACCEPTATE}",
        )

    # 2. Save uploaded file to a temporary file on disk, pastrand extensia reala
    extensie = os.path.splitext(file.filename)[1]
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=extensie) as temp_audio:
            shutil.copyfileobj(file.file, temp_audio)
            temp_audio_path = temp_audio.name
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to save temporary audio file: {e}")

    # 3. Define the blocking transcription task
    def run_whisper_transcription(path: str):
        result = model.transcribe(
            path,
            language="ro",
            initial_prompt="Claseaza informatiile importante",
            fp16=(device == "mps"),
        )
        return result.get("text", "")

    # 4. Run CPU/GPU heavy transcription in a separate thread to prevent blocking FastAPI
    try:
        text = await run_in_threadpool(run_whisper_transcription, temp_audio_path)
    finally:
        # Clean up the temporary file after processing
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    # 5. Extract structured data via LLM (also run in threadpool, e blocking)
    raspuns_llm = await run_in_threadpool(extrage_json, text)

    try:
        date_structurate = json.loads(raspuns_llm)
        date_structurate = completeaza_valori_lipsa(date_structurate)
    except json.JSONDecodeError:
        date_structurate = None

    # 6. Return JSON response with the transcription and extracted data
    return {
        "filename": file.filename,
        "transcription": text,
        "date_structurate": date_structurate,
    }