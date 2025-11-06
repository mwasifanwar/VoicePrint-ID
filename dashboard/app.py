# dashboard/app.py
from flask import Flask, render_template, request, jsonify, Response
import json
import numpy as np
import librosa
import tempfile
import os

def create_app():
    app = Flask(__name__)
    app.config['SECRET_KEY'] = 'voiceprint_id_secret_key_mwasifanwar'
    
    from voiceprint_id.core.speaker_recognizer import SpeakerRecognizer
    from voiceprint_id.core.emotion_detector import EmotionDetector
    from voiceprint_id.core.language_detector import LanguageDetector
    from voiceprint_id.core.anti_spoofing import AntiSpoofing
    from voiceprint_id.core.voice_enhancer import VoiceEnhancer
    
    speaker_recognizer = SpeakerRecognizer()
    emotion_detector = EmotionDetector()
    language_detector = LanguageDetector()
    anti_spoofing = AntiSpoofing()
    voice_enhancer = VoiceEnhancer()
    
    @app.route('/')
    def index():
        return render_template('index.html')
    
    @app.route('/api/analyze_audio', methods=['POST'])
    def analyze_audio():
        try:
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400
            
            audio_file = request.files['audio']
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                audio_file.save(temp_file.name)
                temp_path = temp_file.name
            
            analysis_type = request.form.get('analysis_type', 'all')
            
            results = {}
            
            if analysis_type in ['all', 'speaker']:
                speaker_id, speaker_conf = speaker_recognizer.identify_speaker(temp_path)
                results['speaker'] = {
                    'identified': speaker_id,
                    'confidence': float(speaker_conf) if speaker_conf else 0.0
                }
            
            if analysis_type in ['all', 'emotion']:
                emotion, emotion_conf = emotion_detector.detect_emotion(temp_path)
                results['emotion'] = {
                    'detected': emotion,
                    'confidence': float(emotion_conf) if emotion_conf else 0.0
                }
            
            if analysis_type in ['all', 'language']:
                language, language_conf = language_detector.detect_language(temp_path)
                results['language'] = {
                    'detected': language,
                    'confidence': float(language_conf) if language_conf else 0.0
                }
            
            if analysis_type in ['all', 'spoof']:
                is_real, spoof_conf = anti_spoofing.detect_spoof(temp_path)
                results['spoof_detection'] = {
                    'is_real': is_real,
                    'confidence': float(spoof_conf) if spoof_conf else 0.0
                }
            
            os.unlink(temp_path)
            
            return jsonify({
                'success': True,
                'results': results
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/register_speaker', methods=['POST'])
    def register_speaker():
        try:
            speaker_id = request.form.get('speaker_id')
            audio_files = request.files.getlist('audio_files')
            
            if not speaker_id or not audio_files:
                return jsonify({'error': 'Speaker ID and audio files are required'}), 400
            
            temp_paths = []
            for audio_file in audio_files:
                temp_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
                audio_file.save(temp_file.name)
                temp_paths.append(temp_file.name)
            
            success = speaker_recognizer.register_speaker(speaker_id, temp_paths)
            
            for temp_path in temp_paths:
                os.unlink(temp_path)
            
            return jsonify({
                'success': success,
                'speaker_id': speaker_id,
                'message': 'Speaker registered successfully' if success else 'Failed to register speaker'
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    @app.route('/api/enhance_audio', methods=['POST'])
    def enhance_audio():
        try:
            if 'audio' not in request.files:
                return jsonify({'error': 'No audio file provided'}), 400
            
            audio_file = request.files['audio']
            
            with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp_file:
                audio_file.save(temp_file.name)
                temp_path = temp_file.name
            
            output_path = temp_path + "_enhanced.wav"
            enhanced_audio, sr = voice_enhancer.enhance_audio(temp_path, output_path)
            
            os.unlink(temp_path)
            
            return jsonify({
                'success': True,
                'enhanced_audio_path': output_path,
                'sample_rate': sr
            })
            
        except Exception as e:
            return jsonify({'error': str(e)}), 500
    
    return app

if __name__ == '__main__':
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)