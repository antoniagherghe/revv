import os
import shutil
import tempfile
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.concurrency import run_in_threadpool
import whisper

app = FastAPI(title="Audio Transcription API")

# Extensiile permise
ALLOWED_EXTENSIONS = {".mp3", ".wav"}

# Global variable to hold the loaded model instance
model = None

@app.on_event("startup")
def load_model():
    """Load the Whisper model once when the FastAPI server starts."""
    global model
    print("Se încarcă modelul Whisper în memorie...")
    # Loading the model once globally saves time on every request
    model = whisper.load_model("turbo")
    print("Modelul Whisper a fost încărcat cu succes!")

@app.post("/transcribe/")
async def transcribe_audio(file: UploadFile = File(...)):
    # 1. Validate file extension (.mp3 sau .wav)
    file_ext = os.path.splitext(file.filename)[1].lower()
    if file_ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400, 
            detail=f"Only {', '.join(ALLOWED_EXTENSIONS)} files are supported."
        )

    # 2. Save uploaded file to a temporary file on disk with its dynamic extension
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_ext) as temp_audio:
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
            fp16=False,
        )
        return result.get("text", "")

    # 4. Run CPU/GPU heavy transcription in a separate thread to prevent blocking FastAPI
    try:
        text = await run_in_threadpool(run_whisper_transcription, temp_audio_path)
    finally:
        # Clean up the temporary file after processing
        if os.path.exists(temp_audio_path):
            os.remove(temp_audio_path)

    # 5. Return JSON response with the transcribed text
    return {
        "filename": file.filename,
        "transcription": text
    }