# tests/test_speaker_recognizer.py
import unittest
import numpy as np
import os
import tempfile
import soundfile as sf

class TestSpeakerRecognizer(unittest.TestCase):
    def setUp(self):
        from voiceprint_id.core.speaker_recognizer import SpeakerRecognizer
        self.recognizer = SpeakerRecognizer()
        
        self.test_audio, self.sr = self.generate_test_audio()
        
    def generate_test_audio(self, duration=3.0, sr=16000):
        t = np.linspace(0, duration, int(sr * duration))
        audio = 0.5 * np.sin(2 * np.pi * 440 * t)
        audio += 0.3 * np.sin(2 * np.pi * 880 * t)
        audio += 0.2 * np.random.normal(0, 0.1, len(t))
        return audio, sr
    
    def test_feature_extraction(self):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, self.test_audio, self.sr)
            
            features = self.recognizer.extract_features(temp_file.name)
            self.assertIsNotNone(features)
            self.assertEqual(features.shape, (1, 40, 300, 1))
            
            os.unlink(temp_file.name)
    
    def test_embedding_creation(self):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, self.test_audio, self.sr)
            
            embedding = self.recognizer.create_embedding(temp_file.name)
            self.assertIsNotNone(embedding)
            self.assertEqual(embedding.shape, (256,))
            
            os.unlink(temp_file.name)
    
    def test_speaker_registration(self):
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
            sf.write(temp_file.name, self.test_audio, self.sr)
            
            success = self.recognizer.register_speaker("test_speaker", [temp_file.name])
            self.assertTrue(success)
            self.assertIn("test_speaker", self.recognizer.speaker_database)
            
            os.unlink(temp_file.name)

if __name__ == '__main__':
    unittest.main()