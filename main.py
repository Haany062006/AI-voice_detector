import base64
import json
import re
from fastapi import FastAPI, Header, HTTPException
from pydantic import BaseModel
import google.generativeai as genai

# --- ADD THIS IMPORT ---
from google.generativeai.types import Part

genai.configure(api_key="AIzaSyALpq_FRZcYlsZp1dSC5nUSa0QpQBMnE8I")

# In 2026, ensure you're using the correct model name for audio
model = genai.GenerativeModel('gemini-2.5-flash')

MY_SECRET_KEY = "sk_test_123456789" 

app = FastAPI()

class VoiceRequest(BaseModel):
    language: str
    audioFormat: str
    audioBase64: str

@app.post("/api/voice-detection")
async def detect_voice(request: VoiceRequest, x_api_key: str = Header(None)):
    if x_api_key != MY_SECRET_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")

    try:
        # 1. Clean and Decode
        b64_str = request.audioBase64.strip()
        audio_bytes = base64.b64decode(b64_str)

        # 2. Create the Audio Part
        # This tells Gemini: "This is an audio file, treat it as input"
        audio_part = {
            "mime_type": f"audio/{request.audioFormat}",
            "data": audio_bytes
        }

        prompt = (
            f"Analyze this {request.language} audio. Is it an AI-generated voice or a real human? "
            "Return ONLY a JSON object: { \"classification\": \"AI_GENERATED\"/\"HUMAN\", "
            "\"confidenceScore\": 0.0-1.0, \"explanation\": \"...\" }"
        )

        # 3. Call the model with the correct list format
        # IMPORTANT: The prompt and the audio_part must be in the same list
        response = model.generate_content([prompt, audio_part])

        # 4. Clean the response text (remove ```json wrappers)
        clean_text = re.sub(r'```json|```', '', response.text).strip()
        result = json.loads(clean_text)

        return {
            "status": "success",
            "language": request.language,
            **result
        }

    except Exception as e:
        return {"status": "error", "message": str(e)}
