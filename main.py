import numpy as np
import librosa
import base64
import io
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import uvicorn

# 1. SETUP GOOGLE AI STUDIO (Gemini)
# PASTE YOUR ACTUAL KEY HERE
genai.configure(api_key="YOUR_ACTUAL_GOOGLE_STUDIO_KEY")

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
    if len(y) == 0: return False
    zero_ratio = np.sum(np.abs(y) < 0.00001) / len(y)
    return zero_ratio > 0.04

@app.post("/classify")
async def classify_audio(data: AudioData, api_key: str = Depends(verify_api_key)):
    try:
        # Clean Base64 string (removes data:audio/mp3;base64, prefix if present)
        raw_b64 = data.audio_base64_format
        if "," in raw_b64:
            raw_b64 = raw_b64.split(",")[-1]
            
        audio_bytes = base64.b64decode(raw_b64)
        
        # A. MATH CHECK (Local Analysis)
        # Using 16000Hz as standard for faster processing
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=16000)
        is_ai_math = forensic_math_check(y)
        
        # B. GEMINI CHECK (AI Analysis)
        model = genai.GenerativeModel("gemini-1.5-flash")
        response = model.generate_content([
            "Analyze this audio. Is this an AI-generated voice? Answer with 'Verdict: AI' or 'Verdict: HUMAN' and explain why.",
            {"mime_type": "audio/mp3", "data": audio_bytes}
        ])
        
        gemini_opinion = response.text
        
        return {
            "status": "success",
            "verdict": "AI" if (is_ai_math or "VERDICT: AI" in gemini_opinion.upper()) else "HUMAN",
            "details": {
                "math_detected_ai": bool(is_ai_math),
                "gemini_analysis": gemini_opinion
            }
        }
    except Exception as e:
        # This catch-all helps debug what exactly is failing
        raise HTTPException(status_code=400, detail=f"Error: {str(e)}")

@app.get("/")
async def root():
    return {"status": "online", "message": "The API is LIVE. Ensure your test tool points to /classify"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
