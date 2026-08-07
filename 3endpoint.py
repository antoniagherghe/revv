import json
import os
import tempfile
import time
import warnings
from contextlib import asynccontextmanager
from typing import Dict, Any

import ollama
from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import direct MLX Whisper
try:
    import mlx_whisper
except ImportError:
    raise ImportError("Pachetul `mlx-whisper` este obligatoriu pentru această configurație.")

# --- Configurații Fixe ---
MLX_MODEL = "mlx-community/whisper-large-v3-turbo"
LLM_MODEL = "gemma4:e4b"


# --- Lifespan Event pentru Preload ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    print("=== Inițializare și Preload Modele ===")
    
    # 1. Preload MLX Whisper (executăm o transcriere dummy pe un buffer scurt)
    print(f"[1/2] Se încarcă MLX Whisper ({MLX_MODEL})...")
    try:
        import numpy as np
        dummy_audio = np.zeros(16000, dtype=np.float32) # 1 secundă de liniște
        mlx_whisper.transcribe(dummy_audio, path_or_hf_repo=MLX_MODEL)
        print(" -> MLX Whisper pregătit în memorie.")
    except Exception as e:
        print(f" -> Avertisment la warmup MLX Whisper: {e}")

    # 2. Preload Gemma în Ollama (trimitem o cerere dummy scurtă pentru a încărca modelul în VRAM)
    print(f"[2/2] Se încarcă modelul LLM ({LLM_MODEL}) în Ollama...")
    try:
        ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": "hi"}])
        print(" -> Modelul LLM pregătit în memorie.")
    except Exception as e:
        print(f" -> Eroare la incărcarea modelului Ollama ({LLM_MODEL}): {e}")

    print("=== Sistemul este pregătit pentru cereri ===")
    yield


app = FastAPI(
    title="Service Auto Audio Extractor API",
    description="API dedicat pentru transcriere cu MLX Whisper și extragere date cu Gemma.",
    version="2.0.0",
    lifespan=lifespan
)


# --- Schemă pentru cererea de extragere din text ---
class TextExtractionRequest(BaseModel):
    text: str


# --- Funcții Helper ---
def run_transcription(file_path: str) -> str:
    result = mlx_whisper.transcribe(
        file_path, 
        path_or_hf_repo=MLX_MODEL, 
        language="ro"
    )
    return result["text"]


def run_llm_extraction(text: str) -> Dict[str, Any]:
    prompt = f"""Extrage din textul urmator informatii despre o programare la service auto.
Raspunde STRICT cu un obiect JSON, fara alt text, fara ```json.
Daca o informatie nu apare in text, pune valoarea "-" (liniuta), nu inventa date.

{{"marca": "", "model": "", "an": null, "culoare": "", "motiv_vizita": "", "kilometri": "", "vin": "", "descriere_extra": ""}}

Text: "{text}"
"""
    raspuns = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    raw_content = raspuns["message"]["content"].strip()
    
    # Curățare eventuale marcaje de codeblock
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
        data = run_llm_extraction(payload.text)
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
async def endpoint_process_audio(file: UploadFile = File(...)):
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
        data = run_llm_extraction(text)
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