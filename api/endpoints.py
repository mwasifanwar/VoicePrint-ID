# api/endpoints.py
from fastapi import APIRouter, UploadFile, File, HTTPException, WebSocket
from pydantic import BaseModel
from typing import List, Dict, Any
import tempfile
import os

router = APIRouter()

class SpeakerRegistrationRequest(BaseModel):
    speaker_id: str
    audio_files: List[str] = []

class SpeakerVerificationRequest(BaseModel):
    speaker_id: str
    audio_file: str

class AnalysisResponse(BaseModel):
    success: bool
    result: Dict[str, Any]
    confidence: float
    processing_time: float

@router.post("/speaker/register", response_model=AnalysisResponse)
async def register_speaker(request: SpeakerRegistrationRequest):
    try:
        from voiceprint_id.core.speaker_recognizer import SpeakerRecognizer
        
        recognizer = SpeakerRecognizer()
        
        success = recognizer.register_speaker(request.speaker_id, request.audio_files)
        
        return AnalysisResponse(
            success=success,
            result={"speaker_id": request.speaker_id, "registered": success},
            confidence=1.0 if success else 0.0,
            processing_time=0.0
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/speaker/identify")
async def identify_speaker(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        from voiceprint_id.core.speaker_recognizer import SpeakerRecognizer
        recognizer = SpeakerRecognizer()
        
        speaker_id, confidence = recognizer.identify_speaker(temp_path)
        
        os.unlink(temp_path)
        
        return {
            "success": True,
            "speaker_id": speaker_id,
            "confidence": confidence,
            "identified": speaker_id is not None
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/speaker/verify")
async def verify_speaker(request: SpeakerVerificationRequest):
    try:
        from voiceprint_id.core.speaker_recognizer import SpeakerRecognizer
        recognizer = SpeakerRecognizer()
        
        is_verified, confidence = recognizer.verify_speaker(
            request.audio_file, request.speaker_id
        )
        
        return {
            "success": True,
            "verified": is_verified,
            "confidence": confidence,
            "speaker_id": request.speaker_id
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/emotion/detect")
async def detect_emotion(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        from voiceprint_id.core.emotion_detector import EmotionDetector
        detector = EmotionDetector()
        
        emotion, confidence = detector.detect_emotion(temp_path)
        
        os.unlink(temp_path)
        
        return {
            "success": True,
            "emotion": emotion,
            "confidence": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/language/detect")
async def detect_language(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        from voiceprint_id.core.language_detector import LanguageDetector
        detector = LanguageDetector()
        
        language, confidence = detector.detect_language(temp_path)
        
        os.unlink(temp_path)
        
        return {
            "success": True,
            "language": language,
            "confidence": confidence
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/spoof/detect")
async def detect_spoof(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        from voiceprint_id.core.anti_spoofing import AntiSpoofing
        detector = AntiSpoofing()
        
        is_real, confidence = detector.detect_spoof(temp_path)
        
        os.unlink(temp_path)
        
        return {
            "success": True,
            "is_real": is_real,
            "confidence": confidence,
            "is_spoof": not is_real
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/enhance/audio")
async def enhance_audio(file: UploadFile = File(...)):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
            content = await file.read()
            temp_file.write(content)
            temp_path = temp_file.name
        
        from voiceprint_id.core.voice_enhancer import VoiceEnhancer
        enhancer = VoiceEnhancer()
        
        output_path = temp_path + "_enhanced.wav"
        enhanced_audio, sr = enhancer.enhance_audio(temp_path, output_path)
        
        os.unlink(temp_path)
        
        return {
            "success": True,
            "enhanced_audio_path": output_path,
            "sample_rate": sr
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.websocket("/ws/real_time")
async def websocket_endpoint(websocket: WebSocket):
    from .websocket_handler import WebSocketHandler
    handler = WebSocketHandler()
    await handler.handle_connection(websocket)