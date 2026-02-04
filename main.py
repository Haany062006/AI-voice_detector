import os
import numpy as np
import librosa
import google.generativeai as genai  # Add this import
from fastapi import FastAPI, HTTPException, Header, Depends
# ... (rest of your imports)

# =========================================================
# 1. YOUR GOOGLE AI STUDIO KEY GOES HERE
# =========================================================
genai.configure(api_key="AIzaSyALpq_FRZcYlsZp1dSC5nUSa0QpQBMnE8I")

# 2. YOUR CUSTOM KEY FOR THE FORM (The x-api-key)
VALID_API_KEY = "my_private_key_123" 

app = FastAPI()

# ... (Keep your AudioData class and verify_api_key function)

async def check_with_gemini(audio_data_base64):
    """Optional: Uses Google Gemini to analyze if the speech content is AI-like"""
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        # Sending the audio to Google for a second opinion
        response = model.generate_content([
            "Analyze this audio. Does the rhythm, breath patterns, and script suggest it is AI generated?",
            {"mime_type": "audio/mp3", "data": audio_data_base64}
        ])
        return response.text
    except Exception:
        return "Gemini analysis unavailable."

@app.post("/classify")
async def classify_audio(data: AudioData, api_key: str = Depends(verify_api_key)):
    # Perform the math analysis first (the code we wrote before)
    math_result = analyze_audio_forensics(data.audio_base64_format)
    
    # Optional: Call Gemini using your Google Key
    # gemini_opinion = await check_with_gemini(data.audio_base64_format)
    
    return math_result
