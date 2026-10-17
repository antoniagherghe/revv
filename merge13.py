from dotenv import load_dotenv
load_dotenv()

import json
import os
import re
import requests
import subprocess
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
ultima_replica_acoperita: int = 0  # cate replici din conversatie_completa au fost deja procesate de Gemma

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


ETICHETE_CAMPURI = {
    "marca": "Marcă",
    "model": "Model",
    "an": "An",
    "culoare": "Culoare",
    "motiv_vizita": "Motiv vizită",
    "kilometri": "Kilometri",
    "vin": "VIN",
    "descriere_extra": "Descriere extra",
}


def construieste_continut_notita(profil: Dict[str, Any]) -> str:
    """Construieste corpul notitei ca HTML simplu (Notes accepta HTML de baza in campul 'body')."""
    linii = [
        f"<b>{ETICHETE_CAMPURI.get(cheie, cheie)}:</b> {valoare}"
        for cheie, valoare in profil.items()
    ]
    return "<br>".join(linii)


def _escape_applescript(s: str) -> str:
    """Scapa ghilimelele si backslash-urile, ca sirul sa fie sigur de introdus intr-un script AppleScript."""
    return s.replace("\\", "\\\\").replace('"', '\\"')


def exporta_in_notes(profil: Dict[str, Any]) -> bool:
    """Creeaza automat o notita noua in aplicatia Notes (macOS), cu profilul clientului.
    Ruleaza un script AppleScript prin osascript - Notes poate porni in fundal daca nu
    era deja deschis, dar nu necesita nicio interactiune manuala din partea utilizatorului.
    La prima rulare, macOS cere o singura data permisiune de automatizare
    (System Settings -> Privacy & Security -> Automation)."""
    titlu = f"Profil client - {time.strftime('%d.%m.%Y %H:%M')}"
    continut_html = construieste_continut_notita(profil)

    titlu_escapat = _escape_applescript(titlu)
    continut_escapat = _escape_applescript(continut_html)

    script = f'''
    tell application "Notes"
        make new note with properties {{name:"{titlu_escapat}", body:"{continut_escapat}"}}
    end tell
    '''

    try:
        subprocess.run(["osascript", "-e", script], check=True, capture_output=True, text=True)
        print(" -> Profil exportat cu succes in Notes.")
        return True
    except Exception as e:
        print(f" -> EROARE la exportul in Notes: {e}")
        return False


GOOGLE_FORM_ID = "1FAIpQLSeVBTKp_p4cTxzXXlEKoeRY3GLd2qnfX1awGUKSaDKSvn7hmw"
GOOGLE_FORM_VIEWFORM_URL = f"https://docs.google.com/forms/d/e/{GOOGLE_FORM_ID}/viewform"
GOOGLE_FORM_RESPONSE_URL = f"https://docs.google.com/forms/d/e/{GOOGLE_FORM_ID}/formResponse"

# Maparea camp-din-profil -> eticheta EXACTA a intrebarii asa cum apare in Google Form.
CAMPURI_FORMULAR_GOOGLE = {
    "model": "Model",
    "culoare": "Culoare",
    "motiv_vizita": "Motiv vizita",
}

# Plasa de siguranta: daca auto-detectarea ID-urilor (mai jos) esueaza vreodata pentru ca
# Google isi schimba structura interna a paginii, se pot pune aici manual cele 3 ID-uri
# obtinute din Google Forms -> meniul cu 3 puncte -> "Get pre-filled link" -> completezi
# fiecare camp cu o valoare de test -> "Get link" -> copiezi numerele de dupa "entry." din URL.
# Exemplu: {"model": "123456789", "culoare": "987654321", "motiv_vizita": "555555555"}
ENTRY_ID_OVERRIDE: Dict[str, str] = {}

_entry_id_cache: Dict[str, str] = None


def _obtine_entry_ids_formular() -> Dict[str, str]:
    """Detecteaza automat ID-urile interne (entry.XXXXX) ale celor 3 intrebari din Google Form,
    citind structura publica a paginii formularului. Rezultatul e cache-uit dupa prima rulare
    reusita, ca sa nu se refaca cererea HTTP la fiecare submisie."""
    global _entry_id_cache
    if _entry_id_cache:
        return _entry_id_cache

    if ENTRY_ID_OVERRIDE:
        _entry_id_cache = dict(ENTRY_ID_OVERRIDE)
        return _entry_id_cache

    try:
        raspuns = requests.get(GOOGLE_FORM_VIEWFORM_URL, timeout=10)
        raspuns.raise_for_status()
        match = re.search(r"FB_PUBLIC_LOAD_DATA_\s*=\s*(\[.*?\]);", raspuns.text, re.DOTALL)
        if not match:
            raise ValueError("Nu am gasit structura FB_PUBLIC_LOAD_DATA_ in pagina formularului.")
        structura = json.loads(match.group(1))
        intrebari = structura[1][1]

        gasite: Dict[str, str] = {}
        for intrebare in intrebari:
            titlu = (intrebare[1] or "").strip()
            for camp, eticheta in CAMPURI_FORMULAR_GOOGLE.items():
                if titlu.lower() == eticheta.lower():
                    entry_id = str(intrebare[4][0][0])
                    gasite[camp] = entry_id

        lipsa = [c for c in CAMPURI_FORMULAR_GOOGLE if c not in gasite]
        if lipsa:
            raise ValueError(f"Nu am gasit ID-uri pentru campurile: {lipsa}.")

        _entry_id_cache = gasite
        return gasite
    except Exception as e:
        print(f" -> EROARE la detectarea automata a campurilor din Google Form: {e}")
        print("    -> Foloseste 'Get pre-filled link' din Google Forms si completeaza ENTRY_ID_OVERRIDE manual.")
        return {}


def trimite_in_google_form(profil: Dict[str, Any]) -> bool:
    """Trimite automat cele 3 campuri relevante (model, culoare, motiv_vizita) catre Google Form,
    printr-un simplu request HTTP catre endpoint-ul de raspunsuri al formularului - fara sa
    deschida vreun browser, la fel de discret ca exportul in Notes."""
    entry_ids = _obtine_entry_ids_formular()
    if not entry_ids:
        return False

    payload = {}
    for camp, entry_id in entry_ids.items():
        valoare = profil.get(camp, "-")
        payload[f"entry.{entry_id}"] = "" if valoare == "-" else valoare

    try:
        raspuns = requests.post(GOOGLE_FORM_RESPONSE_URL, data=payload, timeout=10)
        if raspuns.status_code in (200, 302):
            print(" -> Profil trimis cu succes in Google Form.")
            return True
        print(f" -> EROARE: Google Form a raspuns cu status {raspuns.status_code}.")
        return False
    except Exception as e:
        print(f" -> EROARE la trimiterea in Google Form: {e}")
        return False


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
        "CORECTARE GRAMATICALĂ: textul provine dintr-o transcriere audio automată, care poate "
        "conține greșeli de recunoaștere vocală - cuvinte trunchiate, forme gramaticale greșite "
        "sau diacritice lipsă (ex: 'scas' în loc de 'scăzut', 'servis' în loc de 'service'). "
        "Corectează aceste greșeli EVIDENTE de transcriere/gramatică atunci când apar, dar STRICT "
        "la nivel de ortografie și formă a cuvântului - NU schimba sensul, NU înlocui cu un "
        "sinonim diferit și NU generaliza/categorisi (regula de mai sus, de a păstra cuvintele "
        "exacte ale clientului la 'motiv_vizita', rămâne valabilă - doar corectează gramatical "
        "acele cuvinte, nu le înlocui cu altele). Dacă nu ești sigur ce a vrut să spună clientul, "
        "las-o neschimbată, așa cum a fost transcrisă.\n"
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
    global profil_curent, conversatie_completa, contor_replici, ultima_replica_acoperita

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
    replici_neacoperite = len(conversatie_completa) - ultima_replica_acoperita
    if replici_neacoperite >= CORECTIE_LA_FIECARE and conversatie_completa:
        t1 = time.time()
        text_complet = " ".join(conversatie_completa)
        profil_curent = actualizeaza_profil_sigur(profil_curent, await call_gemma_correction(text_complet))
        ultima_replica_acoperita = len(conversatie_completa)
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
    global profil_curent, conversatie_completa, contor_replici, ultima_replica_acoperita
    profil_curent = dict(PROFIL_GOL)
    conversatie_completa = []
    contor_replici = 0
    ultima_replica_acoperita = 0
    return {"status": "reset"}


@app.post("/finalize-session", summary="Ruleaza Gemma pe replicile ramase neacoperite, la final de sesiune")
async def endpoint_finalize_session():
    global profil_curent, conversatie_completa, contor_replici, ultima_replica_acoperita

    if contor_replici == 0 or not conversatie_completa:
        return {
            "extracted_data": profil_curent,
            "corectat_de_gemma": False,
            "motiv": "nicio replica in sesiune",
            "gemma_seconds": 0.0,
        }

    replici_neacoperite = len(conversatie_completa) - ultima_replica_acoperita

    if replici_neacoperite <= 0:
        trimis_in_google_form = await run_in_threadpool(trimite_in_google_form, profil_curent)
        return {
            "extracted_data": profil_curent,
            "corectat_de_gemma": False,
            "motiv": "deja acoperit de ultima trecere Gemma",
            "gemma_seconds": 0.0,
            "trimis_in_google_form": trimis_in_google_form,
        }

    start = time.time()
    text_complet = " ".join(conversatie_completa)
    profil_curent = actualizeaza_profil_sigur(profil_curent, await call_gemma_correction(text_complet))
    ultima_replica_acoperita = len(conversatie_completa)
    duration = round(time.time() - start, 2)

    if not profil_e_gol(profil_curent):
        trimis_in_google_form = await run_in_threadpool(trimite_in_google_form, profil_curent)
    else:
        trimis_in_google_form = False

    return {
        "extracted_data": profil_curent,
        "corectat_de_gemma": True,
        "replici_verificate_suplimentar": replici_neacoperite,
        "gemma_seconds": duration,
        "trimis_in_google_form": trimis_in_google_form,
    }


@app.post("/export-notes", summary="Test izolat: exporta un profil (JSON) direct in Notes")
async def endpoint_export_notes(payload: Dict[str, Any]):
    succes = await run_in_threadpool(exporta_in_notes, payload)
    return {"exportat_in_notes": succes}


@app.post("/export-google-form", summary="Test izolat: trimite un profil (JSON) direct in Google Form")
async def endpoint_export_google_form(payload: Dict[str, Any]):
    succes = await run_in_threadpool(trimite_in_google_form, payload)
    return {"trimis_in_google_form": succes}
