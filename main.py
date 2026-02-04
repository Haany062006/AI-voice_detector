import numpy as np
import librosa
import base64
import io
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import uvicorn

# 1. SETUP GOOGLE AI STUDIO (Gemini)
# Replace with your actual key
genai.configure(api_key="PASTE_YOUR_KEY_HERE")

# 2. SETUP YOUR FORM KEY (x-api-key)
VALID_API_KEY = "hackathon_key_2026"

app = FastAPI()

class AudioData(BaseModel):
    language: str
    audio_format: str
    audio_base64_format: str

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API Key")
    return x_api_key

@app.post("/classify")
async def classify_audio(data: AudioData, api_key: str = Depends(verify_api_key)):
    try:
        # 1. Clean Base64 Data
        b64_str = data.audio_base64_format.split(",")[-1]
        audio_bytes = base64.b64decode(b64_str)
        
        # 2. Forensic Math (Local)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        zero_ratio = np.sum(np.abs(y) < 0.00001) / len(y)
        
        # 3. Gemini Brain (Cloud)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Analyze if this audio is AI-generated. Answer: 'AI' or 'HUMAN' with a reason.",
            {"mime_type": "audio/mp3", "data": audio_bytes}
        ])
        
        is_ai = zero_ratio > 0.04 or "AI" in response.text.upper()
        
        return {
            "status": "success",
            "verdict": "AI" if is_ai else "HUMAN",
            "details": {"math_score": float(zero_ratio), "gemini_report": response.text}
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"status": "online", "message": "The API is running. Submit POST to /classify"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
