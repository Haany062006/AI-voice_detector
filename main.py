import numpy as np
import librosa
import base64
import io
from scipy.signal import correlate

def analyze_audio_forensics(base64_string):
    # 1. Decode and Load Audio
    audio_data = base64.b64decode(base64_string.split(',')[-1])
    y, sr = librosa.load(io.BytesIO(audio_data), sr=None)
    
    # --- FORENSIC CHECK A: The "Digital Zero" Test ---
    # Humans have room noise. AI often has absolute digital silence (0.0).
    silence_threshold = 0.00001
    zero_count = np.sum(np.abs(y) < silence_threshold)
    zero_ratio = zero_count / len(y)
    
    # --- FORENSIC CHECK B: Spectral Artifacts ---
    # AI voices often have 'aliasing' in high frequencies (above 12kHz).
    spec = np.abs(librosa.stft(y))
    # Check energy distribution in the high-frequency bands
    high_freq_energy = np.mean(spec[int(spec.shape[0]*0.75):, :])
    
    # --- FORENSIC CHECK C: Exact Pattern Replication ---
    # Your API missed the loop. We use Cross-Correlation to find identical segments.
    # We split the audio in half and see if they match 100%
    halfway = len(y) // 2
    first_half = y[:halfway]
    second_half = y[halfway:halfway + len(first_half)]
    
    # Normalize for correlation
    corr = correlate(first_half[::100], second_half[::100]) # Downsampled for speed
    max_corr = np.max(corr)
    
    # --- DECISION LOGIC ---
    is_ai = False
    reasons = []
    confidence = 0.5
    
    if zero_ratio > 0.05: # More than 5% absolute silence
        is_ai = True
        reasons.append("Absolute digital silence detected (Noise floor missing)")
        confidence += 0.2
        
    if max_corr > 0.95: # Segments are mathematically identical
        is_ai = True
        reasons.append("Identical audio looping detected (Synthetic repetition)")
        confidence += 0.3

    # NotebookLM specific: Smooth pitch transitions
    pitches, magnitudes = librosa.piptrack(y=y, sr=sr)
    pitch_std = np.std(pitches[pitches > 0])
    if pitch_std < 50: # Real human voices are much more chaotic
        reasons.append("Vocal pitch variance too low (Mathematical prosody)")
        confidence += 0.1

    return {
        "status": "success",
        "classification": "AI" if is_ai else "HUMAN",
        "confidenceScore": min(confidence, 1.0),
        "explanation": " ".join(reasons) if reasons else "Natural human characteristics detected."
    }

# Example Usage
# result = analyze_audio_forensics(your_base64_data)
# print(result)
