import base64
import json
import re
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# --- 1. CONFIGURATION ---
# It is highly recommended to use an Environment Variable on Render for your API Key
# But for now, we will use your provided key:
genai.configure(api_key="AIzaSyALpq_FRZcYlsZp1dSC5nUSa0QpQBMnE8I")

# Using the stable Gemini 2.5 Flash model for 2026
model = genai.GenerativeModel('gemini-2.5-flash')

# This is the secret key required in the 'x-api-key' header
MY_SECRET_KEY = "sk_test_123456789" 

app = FastAPI()

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.get("/")
async def root():
    return {"message": "AI Voice Detector API is running. Use POST /api/voice-detection"}

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    # --- 2. SECURITY CHECK ---
    if x_api_key != MY_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        # --- 3. AUDIO PROCESSING ---
        # Strip potential whitespace and fix padding
        b64_str = request.audioBase64.strip()
        missing_padding = len(b64_str) % 4
        if missing_padding:
            b64_str += "=" * (4 - missing_padding)
            
        audio_bytes = base64.b64decode(b64_str)

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

        # --- 5. GEMINI MULTIMODAL CALL ---
        # Passing as a list of parts ensures the model processes the binary data
        response = model.generate_content(
            contents=[
                prompt,
                {
                    "mime_type": "audio/mp3",
                    "data": audio_bytes
                }
            ]
        )

        # --- 6. JSON CLEANING & PARSING ---
        # Remove markdown code blocks if the model returns them
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
        return {"status": "error", "message": f"Detection failed: {str(e)}"}
