# inference.py
import argparse
import yaml
import json
from datetime import datetime

from voiceprint_id.core.speaker_recognizer import SpeakerRecognizer
from voiceprint_id.core.emotion_detector import EmotionDetector
from voiceprint_id.core.language_detector import LanguageDetector
from voiceprint_id.core.anti_spoofing import AntiSpoofing
from voiceprint_id.core.voice_enhancer import VoiceEnhancer

def main():
    parser = argparse.ArgumentParser(description='VoicePrint ID Inference')
    parser.add_argument('--audio', type=str, required=True, help='Input audio path')
    parser.add_argument('--analysis', type=str, choices=['all', 'speaker', 'emotion', 'language', 'spoof'], default='all')
    parser.add_argument('--output', type=str, help='Output file path')
    
    args = parser.parse_args()
    
    print("Loading VoicePrint ID models...")
    
    results = {}
    
    if args.analysis in ['all', 'speaker']:
        recognizer = SpeakerRecognizer()
        speaker_id, confidence = recognizer.identify_speaker(args.audio)
        results['speaker'] = {
            'identified': speaker_id,
            'confidence': float(confidence) if confidence else 0.0
        }
        print(f"Speaker: {speaker_id} (Confidence: {confidence:.3f})")
    
    if args.analysis in ['all', 'emotion']:
        detector = EmotionDetector()
        emotion, confidence = detector.detect_emotion(args.audio)
        results['emotion'] = {
            'detected': emotion,
            'confidence': float(confidence) if confidence else 0.0
        }
        print(f"Emotion: {emotion} (Confidence: {confidence:.3f})")
    
    if args.analysis in ['all', 'language']:
        detector = LanguageDetector()
        language, confidence = detector.detect_language(args.audio)
        results['language'] = {
            'detected': language,
            'confidence': float(confidence) if confidence else 0.0
        }
        print(f"Language: {language} (Confidence: {confidence:.3f})")
    
    if args.analysis in ['all', 'spoof']:
        detector = AntiSpoofing()
        is_real, confidence = detector.detect_spoof(args.audio)
        results['spoof_detection'] = {
            'is_real': is_real,
            'confidence': float(confidence) if confidence else 0.0,
            'is_spoof': not is_real
        }
        status = "Real" if is_real else "Spoof"
        print(f"Authenticity: {status} (Confidence: {confidence:.3f})")
    
    if args.output:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'audio_file': args.audio,
            'analysis_type': args.analysis,
            'results': results
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()