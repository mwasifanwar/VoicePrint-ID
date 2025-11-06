# api/websocket_handler.py
from fastapi import WebSocket
import json
import numpy as np
import librosa
import asyncio

class WebSocketHandler:
    def __init__(self):
        self.active_connections = []
        
        from voiceprint_id.core.speaker_recognizer import SpeakerRecognizer
        from voiceprint_id.core.emotion_detector import EmotionDetector
        from voiceprint_id.core.language_detector import LanguageDetector
        from voiceprint_id.core.anti_spoofing import AntiSpoofing
        from voiceprint_id.core.voice_enhancer import VoiceEnhancer
        
        self.speaker_recognizer = SpeakerRecognizer()
        self.emotion_detector = EmotionDetector()
        self.language_detector = LanguageDetector()
        self.anti_spoofing = AntiSpoofing()
        self.voice_enhancer = VoiceEnhancer()
    
    async def handle_connection(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)
        
        try:
            while True:
                data = await websocket.receive_text()
                message = json.loads(data)
                
                if message['type'] == 'audio_chunk':
                    audio_data = np.frombuffer(bytes(message['data']), dtype=np.float32)
                    results = await self.process_real_time_audio(audio_data, message.get('sample_rate', 16000))
                    
                    await websocket.send_json({
                        "type": "analysis_results",
                        "results": results
                    })
                
                elif message['type'] == 'register_speaker':
                    success = self.speaker_recognizer.register_speaker(
                        message['speaker_id'], 
                        message['audio_files']
                    )
                    
                    await websocket.send_json({
                        "type": "registration_result",
                        "success": success,
                        "speaker_id": message['speaker_id']
                    })
        
        except Exception as e:
            print(f"WebSocket error: {e}")
        finally:
            self.active_connections.remove(websocket)
    
    async def process_real_time_audio(self, audio_chunk, sample_rate):
        results = {}
        
        enhanced_audio, sr = self.voice_enhancer.real_time_enhance(audio_chunk, sample_rate)
        
        emotion, emotion_conf = self.emotion_detector.detect_emotion_from_audio(enhanced_audio, sr)
        language, language_conf = self.language_detector.detect_language_from_audio(enhanced_audio, sr)
        is_real, spoof_conf = self.anti_spoofing.detect_spoof_from_audio(enhanced_audio, sr)
        
        results['emotion'] = {
            'detected': emotion,
            'confidence': emotion_conf
        }
        
        results['language'] = {
            'detected': language,
            'confidence': language_conf
        }
        
        results['spoof_detection'] = {
            'is_real': is_real,
            'confidence': spoof_conf
        }
        
        return results
    
    async def broadcast_to_all(self, message: dict):
        disconnected = []
        for connection in self.active_connections:
            try:
                await connection.send_json(message)
            except:
                disconnected.append(connection)
        
        for connection in disconnected:
            self.active_connections.remove(connection)