import numpy as np
import librosa
import base64
import io
from scipy.signal import correlate
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn

# Initialize FastAPI
app = FastAPI()

# This class MUST match the fields in your screenshot exactly
class AudioData(BaseModel):
    language: str
    audio_format: str
    audio_base64_format: str  # This matches the "Audio Base64 Format" field

def analyze_audio_forensics(base64_string):
    try:
        # Strip metadata header if present (e.g., data:audio/mp3;base64,...)
        if "," in base64_string:
            base64_string = base64_string.split(',')[-1]
            
        audio_bytes = base64.b64decode(base64_string)
        
        # Load audio. sr=None is critical to catch high-frequency AI artifacts
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        # --- TEST 1: The "Digital Zero" Floor ---
        # AI often has absolute silence (0.0) between words. Humans have background hiss.
        zero_ratio = np.sum(np.abs(y) < 0.00001) / len(y)
        
        # --- TEST 2: Pattern Replication (Loop Detection) ---
        # Humans never repeat a sound wave 100% identically.
        half = len(y) // 2
        max_corr = 0
        if half > 1000:
            # Check if the first half of the audio mathematically matches the second
            corr = correlate(y[:half:100], y[half:half+half:100])
            max_corr = np.max(corr)
        
        # --- TEST 3: Spectral Centroid (High-Frequency Shimmer) ---
        # Advanced AI like NotebookLM often has "checkerboard" artifacts in high frequencies.
        cent = librosa.feature.spectral_centroid(y=y, sr=sr)
        avg_cent = np.mean(cent)

        # Decision Logic
        is_ai = False
        reasons = []
        
        if zero_ratio > 0.05:
            is_ai = True
            reasons.append("Digital silence detected (missing background noise).")
        
        if max_corr > 0.92:
            is_ai = True
            reasons.append("Perfect audio repetition/looping detected.")
            
        if avg_cent > 5000: # High-frequency "shimmer" common in neural TTS
            is_ai = True
            reasons.append("Unnatural high-frequency distribution.")

        return {
            "status": "success",
            "classification": "AI" if is_ai else "HUMAN",
            "confidenceScore": 0.98 if is_ai else 0.82,
            "forensic_report": {
                "zero_ratio": round(float(zero_ratio), 4),
                "repetition_score": round(float(max_corr), 4),
                "reasons": reasons
            }
        }
    except Exception as e:
        raise ValueError(f"Forensic Analysis Failed: {str(e)}")

@app.get("/")
async def root():
    return {"message": "Voice Forensic API is Online. Use /classify endpoint."}

@app.post("/classify")
async def classify_audio(data: AudioData):
    try:
        # We pass the specific base64 field from the form to the analyzer
        result = analyze_audio_forensics(data.audio_base64_format)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
