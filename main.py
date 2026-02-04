import base64
import json
import re
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# --- 1. CONFIGURATION ---
genai.configure(api_key="AIzaSyALpq_FRZcYlsZp1dSC5nUSa0QpQBMnE8I")

# Stable version for Feb 2026
model = genai.GenerativeModel('gemini-2.5-flash')

MY_SECRET_KEY = "sk_test_123456789" 

app = FastAPI()

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.get("/")
async def root():
    return {"message": "AI Voice Detector is Online"}

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    if x_api_key != MY_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        # Clean and decode the Base64 string
        b64_str = request.audioBase64.strip()
        # Fix padding if necessary
        missing_padding = len(b64_str) % 4
        if missing_padding:
            b64_str += "=" * (4 - missing_padding)
            
        audio_bytes = base64.b64decode(b64_str)

        # The prompt needs to be very specific
        prompt = (
            f"You are a forensic expert. Analyze this {request.language} audio for AI manipulation. "
            "Return ONLY a JSON object with: 'classification' (HUMAN or AI_GENERATED), "
            "'confidenceScore' (0.0 to 1.0), and 'explanation'."
        )

        # Passing the audio data directly as a dictionary (No 'Part' import needed)
        response = model.generate_content([
            prompt,
            {
                "mime_type": "audio/mp3",
                "data": audio_bytes
            }
        ])

        # Extract and clean JSON from the response
        response_text = response.text
        clean_text = re.sub(r'```json|```', '', response_text).strip()
        result = json.loads(clean_text)

        return {
            "status": "success",
            "language": request.language,
            "classification": result.get("classification"),
            "confidenceScore": result.get("confidenceScore"),
            "explanation": result.get("explanation")
        }

    except Exception as e:
        return {"status": "error", "message": f"Server Error: {str(e)}"}
