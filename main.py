import numpy as np
import librosa
import base64
import io
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import uvicorn

# 1. SETUP GOOGLE AI STUDIO (Gemini)
# Replace with your actual key from AI Studio
genai.configure(api_key="YOUR_GOOGLE_AI_STUDIO_KEY_HERE")

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

def forensic_math_check(y):
    # Detects the "Digital Zero" floor (common in NotebookLM)
    zero_ratio = np.sum(np.abs(y) < 0.00001) / len(y)
    return zero_ratio > 0.04

@app.post("/classify")
async def classify_audio(data: AudioData, api_key: str = Depends(verify_api_key)):
    try:
        # Clean Base64 string
        b64_str = data.audio_base64_format.split(",")[-1]
        audio_bytes = base64.b64decode(b64_str)
        
        # A. MATH CHECK (Local Analysis)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        is_ai_math = forensic_math_check(y)
        
        # B. GEMINI CHECK (AI Analysis)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Is this audio AI-generated? Look for robotic pacing and perfect pronunciation.",
            {"mime_type": "audio/mp3", "data": audio_bytes}
        ])
        
        gemini_opinion = response.text
        
        return {
            "status": "success",
            "verdict": "AI" if (is_ai_math or "AI" in gemini_opinion.upper()) else "HUMAN",
            "details": {
                "math_detected_ai": is_ai_math,
                "gemini_analysis": gemini_opinion
            }
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"status": "online", "message": "POST to /classify"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
