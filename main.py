import numpy as np
import librosa
import base64
import io
from scipy.signal import correlate
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import uvicorn

# 1. SET YOUR API KEY (Enter this exactly in the "x-api-key" field in your form)
VALID_API_KEY = "hackathon_key_2026" 

app = FastAPI()

# 2. MATCH THE FORM FIELDS FROM YOUR SCREENSHOT
class AudioData(BaseModel):
    language: str
    audio_format: str
    audio_base64_format: str

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid x-api-key")
    return x_api_key

def analyze_audio_forensics(base64_string):
    try:
        # Clean the base64 string
        if "," in base64_string:
            base64_string = base64_string.split(',')[-1]
        audio_bytes = base64.b64decode(base64_string)
        
        # Load audio for mathematical analysis
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        # TEST A: Digital Silence (AI often has absolute zero volume between words)
        zero_ratio = np.sum(np.abs(y) < 0.00001) / len(y)
        
        # TEST B: Repetition Detection (Mathematical loops impossible for humans)
        half = len(y) // 2
        max_corr = 0
        if half > 1000:
            corr = correlate(y[:half:100], y[half:half+half:100])
            max_corr = np.max(corr)

        # Flag as AI if thresholds are met
        is_ai = zero_ratio > 0.04 or max_corr > 0.90
        
        return {
            "status": "success",
            "classification": "AI" if is_ai else "HUMAN",
            "confidenceScore": 0.98 if is_ai else 0.85
        }
    except Exception as e:
        raise ValueError(f"Analysis failed: {str(e)}")

# 3. THIS IS THE ENDPOINT (Ensure your URL ends with /classify)
@app.post("/classify")
async def classify_audio(data: AudioData, api_key: str = Depends(verify_api_key)):
    try:
        return analyze_audio_forensics(data.audio_base64_format)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def health():
    return {"status": "online", "message": "Use /classify for POST requests"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
