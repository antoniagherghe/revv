from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
import tempfile
import time
import warnings
from contextlib import asynccontextmanager
from typing import Dict, Any, List

import ollama
import torch
import whisper
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.concurrency import run_in_threadpool
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

warnings.filterwarnings("ignore", category=RuntimeWarning)

# --- Configurații Fixe ---
WHISPER_MODEL_NAME = "turbo"
GEMMA_LOCAL_MODEL = "gemma4:e4b"

CORECTIE_LA_FIECARE = 5  # dupa cate replici noi ruleaza Gemma


class AutoAppointmentSchema(BaseModel):
    marca: str = Field(description="Marca mașinii (ex: BMW, Audi) sau '-' dacă nu este menționată")
    model: str = Field(description="Modelul mașinii (ex: X3, A4) sau '-' dacă nu este menționat")
    an: str = Field(description="Anul de fabricație sau '-' dacă nu există")
    culoare: str = Field(description="Culoarea vehiculului sau '-' dacă nu există")
    motiv_vizita: str = Field(description="Motivul principal/problema semnalată sau '-'")
    kilometri: str = Field(description="Rulajul în kilometri sau '-' dacă nu există")
    vin: str = Field(description="Seria de șasiu (VIN) sau '-' dacă nu există")
    descriere_extra: str = Field(description="Alte detalii sau observații sau '-'")


REQUIRED_KEYS = list(AutoAppointmentSchema.model_fields.keys())
PROFIL_GOL = {cheie: "-" for cheie in REQUIRED_KEYS}

VALORI_NEGATIVE = {
    "nu", "nu stiu", "nustiu", "nush", "nu am", "n-am", "habar n-am",
    "habar nu am", "nu are", "nush ce sa zic", "poate", "cred ca nu",
    "da", "nu cred", "nu conteaza",
}

CAMPURI_FACTUALE = ["marca", "model", "an", "culoare", "kilometri", "vin"]
CAMPURI_VALIDATE_STRICT = ["marca", "model", "culoare", "vin"]


# --- Stare server (Sesiune unică pe recepție) ---
profil_curent: Dict[str, Any] = dict(PROFIL_GOL)
conversatie_completa: List[str] = []
contor_replici: int = 0

# --- Stare model Whisper (incarcat o singura data la pornire) ---
whisper_model = None
whisper_device = None


# --- Lifespan Event pentru Preload ---
@asynccontextmanager
async def lifespan(app: FastAPI):
    global whisper_model, whisper_device

    print("=== Inițializare și Preload Modele ===")

    print(f"[1/2] Se încarcă Whisper ({WHISPER_MODEL_NAME})...")
    try:
        whisper_device = "mps" if torch.backends.mps.is_available() else "cpu"
        whisper_model = whisper.load_model(WHISPER_MODEL_NAME, device=whisper_device)
        print(f" -> Whisper pregătit în memorie (device: {whisper_device}).")
    except Exception as e:
        print(f" -> Eroare la încărcarea Whisper: {e}")

    print(f"[2/2] Se încarcă modelul Gemma local ({GEMMA_LOCAL_MODEL}) în VRAM...")
    try:
        ollama.chat(model=GEMMA_LOCAL_MODEL, messages=[{"role": "user", "content": "hi"}], keep_alive=-1)
        print(" -> Gemma local pregătit și blocat în VRAM.")
    except Exception as e:
        print(f" -> Eroare la încărcarea modelului Gemma ({GEMMA_LOCAL_MODEL}): {e}")

    print("=== Sistemul este pregătit pentru cereri ===")
    yield


app = FastAPI(
    title="Service Auto Audio Extractor API",
    description="Whisper + Gemma local, Gemma ruleaza periodic cu context complet al conversatiei.",
    version="4.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8080", "http://localhost:3000", "*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


class TextExtractionRequest(BaseModel):
    text: str


def run_transcription(file_path: str) -> str:
    result = whisper_model.transcribe(
        file_path,
        language="ro",
        temperature=0.0,
        no_speech_threshold=0.6,
        logprob_threshold=-1.0,
        compression_ratio_threshold=2.4,
        fp16=(whisper_device == "mps"),
    )
    return result["text"].strip()


def normalize_data(data: dict) -> Dict[str, Any]:
    formatted = {}
    for cheie in REQUIRED_KEYS:
        valoare = data.get(cheie)

        if valoare is None:
            formatted[cheie] = "-"
            continue

        val_str = str(valoare).strip()
        val_lower = val_str.lower()

        if (
            val_lower in ("", "null", "none", "default", "-", "undefined", "n/a")
            or (len(val_str) < 2 and not val_str.isdigit())
        ):
            formatted[cheie] = "-"
        elif cheie in CAMPURI_FACTUALE and val_lower in VALORI_NEGATIVE:
            formatted[cheie] = "-"
        else:
            formatted[cheie] = val_str

    return formatted


def valideaza_grounding(text: str, data: Dict[str, Any]) -> Dict[str, Any]:
    text_normalizat = re.sub(r"[^a-z0-9]", "", text.lower())
    validat = dict(data)
    for cheie in CAMPURI_VALIDATE_STRICT:
        valoare = validat.get(cheie, "-")
        if valoare == "-":
            continue
        valoare_normalizata = re.sub(r"[^a-z0-9]", "", valoare.lower())
        if valoare_normalizata not in text_normalizat:
            validat[cheie] = "-"
    return validat


def profil_e_gol(profil: Dict[str, Any]) -> bool:
    """Adevarat doar daca TOATE campurile sunt '-' - semn ca extragerea a esuat complet."""
    return all(v == "-" for v in profil.values())


def actualizeaza_profil_sigur(profil_vechi: Dict[str, Any], profil_nou: Dict[str, Any]) -> Dict[str, Any]:
    """Gemma are autoritate totala de suprascriere DOAR daca raspunsul ei chiar contine ceva.
    Daca raspunsul nou e complet gol dar cel vechi avea date, pastram profilul vechi -
    un raspuns gol e mai probabil o eroare/esec decat o corectie reala."""
    if profil_e_gol(profil_nou) and not profil_e_gol(profil_vechi):
        print(" -> AVERTISMENT: Gemma a raspuns complet gol, pastrez profilul anterior.")
        return profil_vechi
    return profil_nou


async def call_gemma_correction(full_text: str) -> Dict[str, Any]:
    system_prompt = (
        "Ești un asistent de service auto. Analizează întreaga conversație de mai jos și "
        "extrage datele complete ale vehiculului și ale programării.\n"
        "IMPORTANT: conversația conține atât replicile angajatului de la recepție (saluturi, "
        "prezentări, întrebări puse clientului), cât și răspunsurile clientului. "
        "Extrage informații EXCLUSIV din ce a spus/confirmat CLIENTUL. Ignoră complet salutul, "
        "numele angajatului, întrebările puse de recepție și formulele de politețe - nu le pune "
        "în niciun câmp, în special nu în 'descriere_extra'.\n"
        "Pentru 'motiv_vizita', extrage EXACT cuvintele sau expresia folosita de client. NU "
        "categorisi, NU generaliza si NU inlocui cu termeni tehnici generici (ex: 'Service/"
        "Mentenanta', 'Revizie generala') decat daca clientul a folosit chiar acei termeni. "
        "Daca clientul descrie problema confuz sau nu stie exact ce are, scrie EXACT ce a "
        "spus, cuvant cu cuvant - nu interpreta si nu rezuma.\n"
        "Dacă clientul răspunde negativ sau nu știe ('nu', 'nu stiu', 'nu am', 'habar n-am') la o "
        "întrebare despre un câmp anume, folosește '-' pentru acel câmp - NU scrie cuvântul negativ ca valoare.\n"
        "Returnează STRICT un obiect JSON valid, fără marcaje markdown sau texte adiționale, exact în "
        "acest format:\n"
        '{"marca": "", "model": "", "an": "", "culoare": "", "motiv_vizita": "", "kilometri": "", '
        '"vin": "", "descriere_extra": ""}\n'
        "Nu scrie explicatii, comentarii sau observatii proprii in valorile campurilor - "
        "doar informatia extrasa literal din text, sau '-' daca lipseste."
    )
    user_prompt = f'Conversație completă până în prezent: "{full_text}"'

    try:
        client = ollama.AsyncClient()
        raspuns = await client.chat(
            model=GEMMA_LOCAL_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            options={
                "temperature": 0.0,
                "num_ctx": 8192,
                "num_predict": 250,
            },
            keep_alive=-1,
        )
        continut_brut = raspuns["message"]["content"].strip()
        print(f" -> DEBUG Gemma raspuns brut: {continut_brut}")

        # Curatare eventuale marcaje de codeblock + extragere doar a blocului JSON
        cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", continut_brut).strip()
        match = re.search(r"\{.*\}", cleaned, re.DOTALL)
        data = json.loads(match.group(0)) if match else json.loads(cleaned)
    except Exception as e:
        print(f" -> EROARE Gemma: {type(e).__name__} - {e}")
        data = {}

    data = normalize_data(data)
    print(f" -> DEBUG dupa normalize_data: {data}")
    data = valideaza_grounding(full_text, data)
    print(f" -> DEBUG dupa valideaza_grounding: {data}")
    return data


@app.post("/transcribe", summary="1. Transcrie fișierul audio în text")
async def endpoint_transcribe(file: UploadFile = File(...)):
    start_time = time.time()
    extensie = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=extensie) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = await run_in_threadpool(run_transcription, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)

    return {"text": text, "processing_time_seconds": round(time.time() - start_time, 2)}


@app.post("/extract-info", summary="2. Extrage date structurate din text (Gemma, izolat)")
async def endpoint_extract_info(payload: TextExtractionRequest):
    start_time = time.time()
    data = await call_gemma_correction(payload.text)
    return {"data": data, "processing_time_seconds": round(time.time() - start_time, 2)}


@app.post("/process-audio", summary="3. Pipeline complet (Audio -> Text -> Profil live)")
async def endpoint_process_audio(file: UploadFile = File(...)):
    global profil_curent, conversatie_completa, contor_replici

    total_start = time.time()

    t0 = time.time()
    extensie = os.path.splitext(file.filename)[1] or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=extensie) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        text = await run_in_threadpool(run_transcription, tmp_path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
    transcription_time = round(time.time() - t0, 2)

    if text:
        conversatie_completa.append(text)
    contor_replici += 1

    corectat_de_gemma = False
    gemma_time = 0.0
    if contor_replici % CORECTIE_LA_FIECARE == 0 and conversatie_completa:
        t1 = time.time()
        text_complet = " ".join(conversatie_completa)
        profil_curent = actualizeaza_profil_sigur(profil_curent, await call_gemma_correction(text_complet))
        corectat_de_gemma = True
        gemma_time = round(time.time() - t1, 2)

    return {
        "transcription_text": text,
        "extracted_data": profil_curent,
        "corectat_de_gemma": corectat_de_gemma,
        "numar_replica": contor_replici,
        "durations": {
            "transcription_seconds": transcription_time,
            "gemma_seconds": gemma_time,
            "total_seconds": round(time.time() - total_start, 2),
        },
    }


@app.post("/reset-session", summary="Reseteaza profilul si contorul pentru un client nou")
async def endpoint_reset_session():
    global profil_curent, conversatie_completa, contor_replici
    profil_curent = dict(PROFIL_GOL)
    conversatie_completa = []
    contor_replici = 0
    return {"status": "reset"}


@app.post("/finalize-session", summary="Ruleaza Gemma pe replicile ramase neacoperite, la final de sesiune")
async def endpoint_finalize_session():
    global profil_curent, conversatie_completa, contor_replici

    if contor_replici == 0 or not conversatie_completa:
        return {
            "extracted_data": profil_curent,
            "corectat_de_gemma": False,
            "motiv": "nicio replica in sesiune",
            "gemma_seconds": 0.0,
        }

    if contor_replici % CORECTIE_LA_FIECARE == 0:
        return {
            "extracted_data": profil_curent,
            "corectat_de_gemma": False,
            "motiv": "deja acoperit de ultima trecere Gemma",
            "gemma_seconds": 0.0,
        }

    start = time.time()
    text_complet = " ".join(conversatie_completa)
    profil_curent = actualizeaza_profil_sigur(profil_curent, await call_gemma_correction(text_complet))
    duration = round(time.time() - start, 2)

    replici_neacoperite = contor_replici % CORECTIE_LA_FIECARE

    return {
        "extracted_data": profil_curent,
        "corectat_de_gemma": True,
        "replici_verificate_suplimentar": replici_neacoperite,
        "gemma_seconds": duration,
    }
