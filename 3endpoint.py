import json
import os
import tempfile
import time
import warnings
from typing import Dict, Any

import ollama
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import whisper

warnings.filterwarnings("ignore", category=RuntimeWarning)

app = FastAPI(
    title="Service Auto Audio Extractor API",
    description="API pentru transcriere audio și extragere de date structurate despre programări auto.",
    version="1.0.0",
)

# Încărcăm modelul Whisper o singură dată la start pentru performanță
whisper_model = whisper.load_model("turbo")
DEFAULT_LLM_MODEL = "gemma4:e4b"


# --- Schemă pentru cererea de extragere din text ---
class TextExtractionRequest(BaseModel):
    text: str
    llm_model: str = DEFAULT_LLM_MODEL


# --- Funcții Helper ---
def run_transcription(file_path: str) -> str:
    result = whisper_model.transcribe(file_path, language="ro", fp16=False)
    return result["text"]


def run_llm_extraction(text: str, model_name: str) -> Dict[str, Any]:
    prompt = f"""Extrage din textul urmator informatii despre o programare la service auto.
Raspunde STRICT cu un obiect JSON, fara alt text, fara ```json.
Daca o informatie nu apare in text, pune valoarea "-" (liniuta), nu inventa date.

{{"marca": "", "model": "", "an": null, "culoare": "", "motiv_vizita": "", "kilometri": "", "vin": "", "descriere_extra": ""}}

Text: "{text}"
"""
    raspuns = ollama.chat(model=model_name, messages=[{"role": "user", "content": prompt}])
    raw_content = raspuns["message"]["content"].strip()
    
    # Curățare eventuale marcaje de codeblock dacă LLM-ul le adaugă totuși
    if raw_content.startswith("```json"):
        raw_content = raw_content[7:]
    if raw_content.startswith("```"):
        raw_content = raw_content[3:]
    if raw_content.endswith("```"):
        raw_content = raw_content[:-3]
        
    data = json.loads(raw_content.strip())
    
    # Completează valorile lipsă
    for cheie, valoare in data.items():
        if valoare is None or (isinstance(valoare, str) and valoare.strip() == ""):
            data[cheie] = "-"
            
    return data


# --- Endpoint 1: Audio -> Text ---
@app.post("/transcribe", summary="1. Transcrie fișierul audio în text")
async def endpoint_transcribe(file: UploadFile = File(...)):
    start_time = time.time()
    
    # Salvăm temporar fișierul primit pe disc pentru a fi citit de Whisper
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = run_transcription(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    duration = round(time.time() - start_time, 2)
    return {
        "text": text,
        "processing_time_seconds": duration
    }


# --- Endpoint 2: Text -> JSON Structurat ---
@app.post("/extract-info", summary="2. Extrage date structurate din text")
async def endpoint_extract_info(payload: TextExtractionRequest):
    start_time = time.time()
    
    try:
        data = run_llm_extraction(payload.text, payload.llm_model)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Răspunsul generat de LLM nu a fost un JSON valid.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    duration = round(time.time() - start_time, 2)
    return {
        "data": data,
        "processing_time_seconds": duration
    }


# --- Endpoint 3: Audio -> JSON Structurat (Pipeline Complet) ---
@app.post("/process-audio", summary="3. Pipeline complet (Audio -> Text -> JSON)")
async def endpoint_process_audio(
    file: UploadFile = File(...), 
    llm_model: str = DEFAULT_LLM_MODEL
):
    total_start = time.time()
    
    # Pasul 1: Transcriere
    t0 = time.time()
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = run_transcription(tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
            
    transcription_time = round(time.time() - t0, 2)

    # Pasul 2: Extragere JSON
    t1 = time.time()
    try:
        data = run_llm_extraction(text, llm_model)
    except json.JSONDecodeError:
        raise HTTPException(status_code=422, detail="Răspunsul generat de LLM nu a fost un JSON valid.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    llm_time = round(time.time() - t1, 2)
    total_time = round(time.time() - total_start, 2)

    return {
        "transcription_text": text,
        "extracted_data": data,
        "durations": {
            "transcription_seconds": transcription_time,
            "llm_seconds": llm_time,
            "total_seconds": total_time
        }
    }