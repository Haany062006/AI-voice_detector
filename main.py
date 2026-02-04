import os
import base64
import io
import numpy as np
import librosa
from fastapi import FastAPI, HTTPException, Header, Depends
from pydantic import BaseModel
import google.generativeai as genai
import uvicorn

# 1. API KEYS
# Paste your Google AI Studio key here or set it in Render Environment Variables
GOOGLE_API_KEY = "AIzaSyALpq_FRZcYlsZp1dSC5nUSa0QpQBMnE8I"
genai.configure(api_key=GOOGLE_API_KEY)

# This is the x-api-key for your form
VALID_API_KEY = "hackathon_key_2026" 

app = FastAPI()

class AudioData(BaseModel):
    language: str
    audio_format: str
    audio_base64_format: str

async def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != VALID_API_KEY:
        raise HTTPException(status_code=401, detail="Invalid x-api-key")
    return x_api_key

async def gemini_voice_analysis(audio_b64):
    """Uses Gemini 1.5 Flash to detect AI speech patterns/scripts"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Prepare the audio part for Gemini
        audio_part = {"mime_type": "audio/mp3", "data": audio_b64}
        
        prompt = (
            "Analyze this audio. Is this an AI-generated voice? "
            "Look for unnatural breathing, robotic pacing, and 'perfect' pronunciation. "
            "Reply with 'AI' or 'HUMAN' and a short reason."
        )
        
        response = model.generate_content([prompt, audio_part])
        return response.text
    except Exception as e:
        return f"Gemini Error: {str(e)}"

@app.post("/classify")
async def classify_audio(data: AudioData, api_key: str = Depends(verify_api_key)):
    try:
        # Clean the string if it has the "data:audio/mp3;base64," prefix
        b64_data = data.audio_base64_format.split(',')[-1]
        
        # 1. Mathematical Forensic Check (Local)
        audio_bytes = base64.b64decode(b64_data)
        y, sr = librosa.load(io.BytesIO(audio_bytes), sr=None)
        zero_ratio = np.sum(np.abs(y) < 0.00001) / len(y)
        
        # 2. AI Model Check (Cloud via Google Key)
        gemini_report = await gemini_voice_analysis(b64_data)
        
        is_ai_math = zero_ratio > 0.04
        
        return {
            "status": "success",
            "forensic_classification": "AI" if is_ai_math else "HUMAN",
            "gemini_analysis": gemini_report,
            "final_verdict": "Likely AI" if (is_ai_math or "AI" in gemini_report.upper()) else "Likely Human"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/")
async def root():
    return {"status": "online", "endpoint": "/classify"}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
