import json
import time
import warnings

import ollama
import whisper

warnings.filterwarnings("ignore", category=RuntimeWarning)

# ---- AICI SCHIMBI CE MODEL LLM TESTEZI ----
#variante: "gemma4:e4b", "qwen2.5:7b", "llama3.1:8b"
LLM_MODEL = "gemma4:e4b"


# ---- Pasul 1: transcriere audio -> text ----
def transcribe(audio_file):
    model = whisper.load_model("turbo")
    result = model.transcribe(audio_file, language="ro", fp16=False)
    return result["text"]


# ---- Pasul 2: text -> JSON structurat, folosind LLM-ul ales ----
def extrage_json(text):
    prompt = f"""Extrage din textul urmator informatii despre o programare la service auto.
Raspunde STRICT cu un obiect JSON, fara alt text, fara ```json.
Daca o informatie nu apare in text, pune valoarea "-" (liniuta), nu inventa date.

{{"marca": "", "model": "", "an": null, "culoare": "", "motiv_vizita": "", "kilometri": "", "vin": "", "descriere_extra": ""}}

Text: "{text}"
"""
    raspuns = ollama.chat(model=LLM_MODEL, messages=[{"role": "user", "content": prompt}])
    return raspuns["message"]["content"]


# ---- Inlocuieste orice valoare goala/lipsa cu "-" ----
def completeaza_valori_lipsa(date: dict) -> dict:
    for cheie, valoare in date.items():
        if valoare is None or (isinstance(valoare, str) and valoare.strip() == ""):
            date[cheie] = "-"
    return date


# ---- Rulare ----
with open("config.json", "r", encoding="utf-8") as f:
    config = json.load(f)

audio_file = config.get("audio_file", "recording.wav")

print("Transcriu audio...")
text = transcribe(audio_file)
print(f"\nText transcris:\n{text}")

print(f"\nExtrag date cu modelul '{LLM_MODEL}'...")
start_llm = time.time()
raspuns_json = extrage_json(text)
timp_llm = round(time.time() - start_llm, 2)
print(f"\nRaspuns LLM:\n{raspuns_json}")
print(f"\nTimp extragere JSON: {timp_llm}s")

# Salvam rezultatul intr-un fisier json
try:
    date = json.loads(raspuns_json)
    date = completeaza_valori_lipsa(date)
    with open("rezultat.json", "w", encoding="utf-8") as f:
        json.dump(date, f, ensure_ascii=False, indent=2)
    print("\nSalvat cu succes in rezultat.json")
except json.JSONDecodeError:
    print("\nATENTIE: raspunsul nu a fost JSON valid, nu s-a salvat nimic.")