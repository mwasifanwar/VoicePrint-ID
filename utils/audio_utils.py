# utils/audio_utils.py
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr

class AudioUtils:
    def __init__(self, target_sr=16000):
        self.target_sr = target_sr
    
    def convert_sample_rate(self, audio, original_sr):
        if original_sr != self.target_sr:
            return librosa.resample(audio, orig_sr=original_sr, target_sr=self.target_sr)
        return audio
    
    def trim_silence(self, audio, top_db=20):
        return librosa.effects.trim(audio, top_db=top_db)[0]
    
    def normalize_audio(self, audio):
        return librosa.util.normalize(audio)
    
    def compute_energy(self, audio):
        return np.sum(audio ** 2) / len(audio)
    
    def compute_zcr(self, audio):
        return np.mean(librosa.feature.zero_crossing_rate(audio))
    
    def compute_spectral_centroid(self, audio, sr):
        return np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
    
    def compute_spectral_rolloff(self, audio, sr, roll_percent=0.85):
        return np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr, roll_percent=roll_percent))
    
    def compute_mfcc_stats(self, audio, sr, n_mfcc=13):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        return {
            'mean': np.mean(mfccs, axis=1),
            'std': np.std(mfccs, axis=1),
            'max': np.max(mfccs, axis=1),
            'min': np.min(mfccs, axis=1)
        }
    
    def detect_voice_activity(self, audio, sr, threshold=0.01):
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2)
            for i in range(0, len(audio)-frame_length, hop_length)
        ])
        
        threshold_energy = np.max(energy) * threshold
        voice_frames = energy > threshold_energy
        
        voice_ratio = np.sum(voice_frames) / len(voice_frames)
        
        return voice_ratio > 0.3
    
    def extract_audio_segments(self, audio, sr, segment_duration=3.0, overlap=0.5):
        segment_length = int(segment_duration * sr)
        overlap_length = int(segment_length * overlap)
        step_length = segment_length - overlap_length
        
        segments = []
        
        for start in range(0, len(audio) - segment_length + 1, step_length):
            end = start + segment_length
            segment = audio[start:end]
            segments.append(segment)
        
        return segments
    
    def merge_audio_segments(self, segments, overlap=0.5):
        segment_length = len(segments[0])
        overlap_length = int(segment_length * overlap)
        step_length = segment_length - overlap_length
        
        total_length = segment_length + (len(segments) - 1) * step_length
        merged_audio = np.zeros(total_length)
        
        for i, segment in enumerate(segments):
            start = i * step_length
            end = start + segment_length
            
            if i == 0:
                merged_audio[start:end] = segment
            else:
                overlap_region = merged_audio[start:start+overlap_length]
                current_overlap = segment[:overlap_length]
                
                weights = np.linspace(0, 1, overlap_length)
                blended_overlap = overlap_region * (1 - weights) + current_overlap * weights
                
                merged_audio[start:start+overlap_length] = blended_overlap
                merged_audio[start+overlap_length:end] = segment[overlap_length:]
        
        return merged_audio