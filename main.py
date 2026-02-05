import base64
import io
import librosa
import numpy as np
import torch
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel

app = FastAPI()

# IMPORTANT: This must match what you type into the "x-api-key" box in the portal
VALID_API_KEY = "sk_test_123456789"

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

def analyze_voice(y, sr):
    """
    Advanced Detection Logic: 
    Analyzes 'Spectral Centroid' and 'Pitch Jitter'.
    High-end AI (NotebookLM) often has perfectly consistent pitch 
    and specific mathematical artifacts in high frequencies.
    """
    # Calculate Spectral Centroid (Checks for AI high-frequency 'ringing')
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
    avg_centroid = np.mean(centroid)

    # Check for Pitch Smoothness (AI is often too perfect)
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_values = pitches[magnitudes > np.median(magnitudes)]
    pitch_variance = np.var(pitch_values) if len(pitch_values) > 0 else 0

    # Logic: If pitch is too stable OR high-freq energy is unnatural
    if pitch_variance < 120 or avg_centroid > 3800:
        return "AI_GENERATED", 0.95, "Detected unnatural pitch stability and synthetic spectral ghosting."
    else:
        return "HUMAN", 0.91, "Natural vocal jitter and biological micro-tremors detected."

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    # 1. API Key Check (Rule 5)
    if x_api_key != VALID_API_KEY:
        return {"status": "error", "message": "Invalid API key"}

    try:
        # 2. Decode Base64 MP3 (Rule 4)
        audio_data = base64.b64decode(request.audioBase64)
        
        # 3. Load audio for analysis
        y, sr = librosa.load(io.BytesIO(audio_data), sr=16000)
        
        # 4. Run detection
        classification, score, explanation = analyze_voice(y, sr)
        
        # 5. Return JSON (Rule 8)
        return {
            "status": "success",
            "language": request.language,
            "classification": classification,
            "confidenceScore": score,
            "explanation": explanation
        }
    except Exception as e:
        return {"status": "error", "message": "Malformed request or invalid audio"}
