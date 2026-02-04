import numpy as np
import librosa
import base64
import io
from scipy.signal import correlate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# 1. THIS IS THE LINE UVICORN IS LOOKING FOR
app = FastAPI()

class AudioData(BaseModel):
    audio_base64: str

def analyze_audio_forensics(base64_string):
    # (The logic we built)
    if "," in base64_string:
        base64_string = base64_string.split(',')[-1]
    
    audio_bytes = base64.b64decode(base64_string)
    y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
    
    # Check for AI 'Zero-Floor'
    zero_ratio = np.sum(np.abs(y) < 0.00001) / len(y)
    
    # Check for Loops (Correlation)
    half = len(y) // 2
    max_corr = 0
    if half > 1000:
        corr = correlate(y[:half:100], y[half:half+half:100])
        max_corr = np.max(corr)

    is_ai = zero_ratio > 0.05 or max_corr > 0.90
    
    return {
        "classification": "AI" if is_ai else "HUMAN",
        "confidence": 0.95 if is_ai else 0.80
    }

# 2. DEFINE THE ENDPOINT
@app.post("/classify")
async def classify_audio(data: AudioData):
    try:
        return analyze_audio_forensics(data.audio_base64)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# 3. ROOT ENDPOINT (Optional - helps you check if the site is up)
@app.get("/")
async def root():
    return {"message": "API is running. Use /classify for detection."}
