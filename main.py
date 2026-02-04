import base64
import json
import re
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# --- 1. CONFIGURATION ---
# Use your AI Studio key
genai.configure(api_key="AIzaSyALpq_FRZcYlsZp1dSC5nUSa0QpQBMnE8I")

# UPDATED: Using Gemini 3 Flash for maximum accuracy and speed
model = genai.GenerativeModel('gemini-1.5-flash-latest')

# This is the key the judges will use to access your API
MY_SECRET_KEY = "sk_test_123456789" 

app = FastAPI()

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    # --- 2. SECURITY CHECK ---
    if x_api_key != MY_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        # --- 3. AUDIO PROCESSING ---
        audio_bytes = base64.b64decode(request.audioBase64)

        # --- 4. FORENSIC PROMPT ---
        prompt = (
            f"You are a forensic audio expert. Analyze this {request.language} voice recording. "
            "Examine it for AI-generated artifacts such as unnatural phonetic transitions, "
            "lack of breathing pauses, or digital spectral noise. "
            "Return ONLY a JSON object with these fields: "
            "classification (string: 'AI_GENERATED' or 'HUMAN'), "
            "confidenceScore (float between 0 and 1), "
            "explanation (string detail why)."
        )

        # Send to Gemini
        response = model.generate_content([
            prompt,
            {'mime_type': 'audio/mp3', 'data': audio_bytes}
        ])

        # --- 5. ROBUST JSON CLEANING ---
        # This removes any markdown formatting (like ```json) Gemini might add
        clean_text = re.sub(r'```json|```', '', response.text).strip()
        result = json.loads(clean_text)

        return {
            "status": "success",
            "language": request.language,
            "classification": result.get("classification"),
            "confidenceScore": result.get("confidenceScore"),
            "explanation": result.get("explanation")
        }

    except Exception as e:
        # Detailed error reporting to help you debug during testing
        return {"status": "error", "message": f"Detection failed: {str(e)}"}

