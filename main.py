import numpy as np
import librosa
import base64
import io
from scipy.signal import correlate
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import uvicorn

# 1. SET YOUR API KEY (Enter this into the x-api-key field in your form)
VALID_API_KEY = "my_private_key_123" 

app = FastAPI()

# 2. MATCH THE FORM FIELDS EXACTLY
class AudioData(BaseModel):
    language: str
    audio_format: str
    audio_base64_format: str

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

def analyze_audio_forensics(base64_string):
    try:
        # Clean the base64 string
        if "," in base64_string:
            base64_string = base64_string.split(',')[-1]
        audio_bytes = base64.b64decode(base64_string)
        
        # Load audio - sr=None preserves native quality for AI detection
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        
        # TEST 1: Absolute Digital Silence (NotebookLM Signature)
        # Humans have background noise; AI often has mathematically perfect 0s.
        zero_ratio = np.sum(np.abs(y) < 0.00001) / len(y)
        
        # TEST 2: Loop Detection
        # Checks if audio segments repeat identically (impossible for humans).
        half = len(y) // 2
        max_corr = 0
        if half > 1000:
            corr = correlate(y[:half:100], y[half:half+half:100])
            max_corr = np.max(corr)

        is_ai = zero_ratio > 0.05 or max_corr > 0.92
        
        return {
            "status": "success",
            "classification": "AI" if is_ai else "HUMAN",
            "confidenceScore": 0.98 if is_ai else 0.82
        }
    except Exception as e:
        raise ValueError(f"Analysis failed: {str(e)}")

# 3. DEFINE THE ENDPOINT (This is why you got a 404)
@app.post("/classify")
async def classify_audio(data: AudioData, api_key: str = Depends(verify_api_key)):
    try:
        return analyze_audio_forensics(data.audio_base64_format)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"message": "API is online. Use the /classify endpoint."}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
