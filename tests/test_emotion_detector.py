# tests/test_emotion_detector.py
import unittest
import numpy as np
import tempfile
import soundfile as sf

class TestEmotionDetector(unittest.TestCase):
    def setUp(self):
        from voiceprint_id.core.emotion_detector import EmotionDetector
        self.detector = EmotionDetector()
        
        self.test_audio, self.sr = self.generate_test_audio()
    
    def generate_test_audio(self, duration=3.0, sr=16000):
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        return audio, sr
    
    def test_emotion_feature_extraction(self):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, self.test_audio, self.sr)
            
            features = self.detector.extract_emotion_features(temp_file.name)
            self.assertIsNotNone(features)
            self.assertEqual(features.shape, (1, 40, 300, 1))
            
            import os
            os.unlink(temp_file.name)
    
    def test_emotion_detection(self):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, self.test_audio, self.sr)
            
            emotion, confidence = self.detector.detect_emotion(temp_file.name)
            self.assertIsNotNone(emotion)
            self.assertIsInstance(confidence, float)
            self.assertGreaterEqual(confidence, 0.0)
            self.assertLessEqual(confidence, 1.0)
            
            import os
            os.unlink(temp_file.name)

if __name__ == '__main__':
    unittest.main()