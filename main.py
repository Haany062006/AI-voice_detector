import base64
import json
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# --- 1. CONFIGURATION ---
# Replace the string below with your key from Step 1
genai.configure(api_key="AIzaSyALpq_FRZcYlsZp1dSC5nUSa0QpQBMnE8I")
model = genai.GenerativeModel('gemini-1.5-flash')

# This is the key the judges will use to access YOUR API
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

        # --- 4. FORENSIC PROMPT (THE WINNING EDGE) ---
        prompt = (
            f"You are a forensic audio expert. Analyze this {request.language} voice. "
            "Determine if it is AI_GENERATED or HUMAN. Focus on 'robotic smoothing' "
            "in regional phonemes and the presence/absence of natural breathing. "
            "Return ONLY a JSON object with: classification, confidenceScore, explanation."
        )

        # Send to Gemini
        response = model.generate_content([
            prompt,
            {'mime_type': 'audio/mp3', 'data': audio_bytes}
        ])

        # --- 5. CLEANING THE OUTPUT ---
        # Gemini sometimes adds markdown; we strip it to get pure JSON
        raw_text = response.text.replace('```json', '').replace('```', '').strip()
        result = json.loads(raw_text)

        return {
            "status": "success",
            "language": request.language,
            "classification": result.get("classification"),
            "confidenceScore": result.get("confidenceScore"),
            "explanation": result.get("explanation")
        }

    except Exception as e:

        return {"status": "error", "message": f"Processing error: {str(e)}"}
