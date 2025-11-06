# core/voice_enhancer.py
import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr
from scipy import signal

class VoiceEnhancer:
    def __init__(self):
        self.sr = 16000
        
    def enhance_audio(self, audio_path, output_path=None):
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            enhanced_audio = self._apply_enhancement_pipeline(audio, sr)
            
            if output_path:
                sf.write(output_path, enhanced_audio, sr)
            
            return enhanced_audio, sr
            
        except Exception as e:
            print(f"Audio enhancement error: {e}")
            return None, None
    
    def _apply_enhancement_pipeline(self, audio, sr):
        audio_enhanced = audio.copy()
        
        audio_enhanced = self._reduce_noise(audio_enhanced, sr)
        audio_enhanced = self._normalize_audio(audio_enhanced)
        audio_enhanced = self._equalize_audio(audio_enhanced, sr)
        audio_enhanced = self._remove_silence(audio_enhanced, sr)
        
        return audio_enhanced
    
    def _reduce_noise(self, audio, sr):
        try:
            reduced_noise = nr.reduce_noise(y=audio, sr=sr, prop_decrease=0.8)
            return reduced_noise
        except:
            return audio
    
    def _normalize_audio(self, audio):
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio_normalized = audio / max_val * 0.9
            return audio_normalized
        return audio
    
    def _equalize_audio(self, audio, sr):
        frequencies = [100, 500, 1000, 2000, 4000, 8000]
        gains = [2, 3, 4, 3, 2, 1]
        
        try:
            sos = signal.iirfilter(4, [frequencies[0], frequencies[-1]], 
                                 rs=20, btype='bandpass',
                                 analog=False, fs=sr, output='sos')
            
            equalized = signal.sosfilt(sos, audio)
            
            for i in range(len(frequencies)-1):
                low_freq = frequencies[i]
                high_freq = frequencies[i+1]
                gain = gains[i]
                
                sos_band = signal.iirfilter(4, [low_freq, high_freq], 
                                          rs=20, btype='bandpass',
                                          analog=False, fs=sr, output='sos')
                
                band_audio = signal.sosfilt(sos_band, audio)
                equalized += band_audio * (gain - 1)
            
            return equalized
            
        except:
            return audio
    
    def _remove_silence(self, audio, sr, threshold=0.02):
        frame_length = int(0.025 * sr)
        hop_length = int(0.010 * sr)
        
        energy = np.array([
            np.sum(audio[i:i+frame_length]**2)
            for i in range(0, len(audio)-frame_length, hop_length)
        ])
        
        threshold_energy = np.max(energy) * threshold
        
        voiced_frames = energy > threshold_energy
        
        if not np.any(voiced_frames):
            return audio
        
        voiced_segments = []
        current_segment = []
        
        for i, is_voiced in enumerate(voiced_frames):
            start_sample = i * hop_length
            end_sample = start_sample + frame_length
            
            if is_voiced:
                current_segment.extend(audio[start_sample:end_sample])
            else:
                if len(current_segment) > 0:
                    voiced_segments.append(current_segment)
                    current_segment = []
        
        if len(current_segment) > 0:
            voiced_segments.append(current_segment)
        
        if len(voiced_segments) == 0:
            return audio
        
        result_audio = np.concatenate(voiced_segments)
        
        return result_audio
    
    def real_time_enhance(self, audio_chunk, sr):
        try:
            if sr != self.sr:
                audio_chunk = librosa.resample(audio_chunk, orig_sr=sr, target_sr=self.sr)
            
            enhanced_chunk = self._apply_enhancement_pipeline(audio_chunk, self.sr)
            
            return enhanced_chunk, self.sr
            
        except Exception as e:
            print(f"Real-time enhancement error: {e}")
            return audio_chunk, sr