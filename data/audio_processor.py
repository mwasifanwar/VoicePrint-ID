# data/audio_processor.py
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr

class AudioProcessor:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
    
    def load_audio(self, file_path, duration=None):
        try:
            audio, sr = librosa.load(file_path, sr=self.target_sr, duration=duration)
            return audio, sr
        except Exception as e:
            print(f"Error loading audio: {e}")
            return None, None
    
    def extract_mfcc(self, audio, sr, n_mfcc=40, n_fft=2048, hop_length=512):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc, 
                                   n_fft=n_fft, hop_length=hop_length)
        return mfccs
    
    def extract_mel_spectrogram(self, audio, sr, n_mels=128, n_fft=2048, hop_length=512):
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr, n_mels=n_mels,
                                                n_fft=n_fft, hop_length=hop_length)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        return mel_spec_db
    
    def extract_chroma_features(self, audio, sr, n_fft=2048, hop_length=512):
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr, n_fft=n_fft, 
                                           hop_length=hop_length)
        return chroma
    
    def extract_spectral_contrast(self, audio, sr, n_fft=2048, hop_length=512):
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr,
                                                            n_fft=n_fft, hop_length=hop_length)
        return spectral_contrast
    
    def extract_tonnetz(self, audio, sr):
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        return tonnetz
    
    def extract_all_features(self, audio, sr):
        features = {}
        
        features['mfcc'] = self.extract_mfcc(audio, sr)
        features['mel_spectrogram'] = self.extract_mel_spectrogram(audio, sr)
        features['chroma'] = self.extract_chroma_features(audio, sr)
        features['spectral_contrast'] = self.extract_spectral_contrast(audio, sr)
        features['tonnetz'] = self.extract_tonnetz(audio, sr)
        
        features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(audio)
        features['spectral_centroid'] = librosa.feature.spectral_centroid(y=audio, sr=sr)
        features['spectral_rolloff'] = librosa.feature.spectral_rolloff(y=audio, sr=sr)
        features['rms_energy'] = librosa.feature.rms(y=audio)
        
        return features
    
    def preprocess_audio(self, audio_path, output_path=None, enhance=True):
        audio, sr = self.load_audio(audio_path)
        if audio is None:
            return None, None
        
        if enhance:
            audio = self.enhance_audio(audio, sr)
        
        if output_path:
            sf.write(output_path, audio, sr)
        
        return audio, sr
    
    def enhance_audio(self, audio, sr):
        audio_enhanced = audio.copy()
        
        audio_enhanced = nr.reduce_noise(y=audio_enhanced, sr=sr, prop_decrease=0.7)
        
        audio_enhanced = self.normalize_audio(audio_enhanced)
        
        return audio_enhanced
    
    def normalize_audio(self, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            return audio / max_val * 0.9
        return audio
    
    def split_audio(self, audio, sr, segment_duration=3.0):
        segment_length = int(segment_duration * sr)
        segments = []
        
        for start in range(0, len(audio), segment_length):
            end = start + segment_length
            if end <= len(audio):
                segment = audio[start:end]
                segments.append(segment)
        
        return segments
    
    def pad_audio(self, audio, target_length):
        if len(audio) >= target_length:
            return audio[:target_length]
        else:
            return np.pad(audio, (0, target_length - len(audio)), mode='constant')