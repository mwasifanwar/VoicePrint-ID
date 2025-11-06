# data/data_augmentation.py
import numpy as np
import librosa
import soundfile as sf

class AudioAugmentation:
    def __init__(self, sr=16000):
        self.sr = sr
    
    def add_noise(self, audio, noise_level=0.005):
        noise = np.random.normal(0, noise_level, audio.shape)
        return audio + noise
    
    def time_shift(self, audio, shift_max=0.2):
        shift = np.random.randint(-int(self.sr * shift_max), int(self.sr * shift_max))
        augmented = np.roll(audio, shift)
        
        if shift > 0:
            augmented[:shift] = 0
        else:
            augmented[shift:] = 0
        
        return augmented
    
    def time_stretch(self, audio, rate=1.0):
        return librosa.effects.time_stretch(audio, rate=rate)
    
    def pitch_shift(self, audio, n_steps=2):
        return librosa.effects.pitch_shift(audio, sr=self.sr, n_steps=n_steps)
    
    def change_speed(self, audio, speed_factor=1.0):
        return librosa.effects.time_stretch(audio, rate=speed_factor)
    
    def change_volume(self, audio, db_change=3.0):
        return audio * (10 ** (db_change / 20))
    
    def apply_room_impulse_response(self, audio, room_size=0.5):
        impulse_response = self._generate_room_impulse(room_size)
        augmented = np.convolve(audio, impulse_response, mode='same')
        return augmented
    
    def _generate_room_impulse(self, room_size):
        length = int(self.sr * room_size)
        impulse = np.random.normal(0, 0.1, length)
        impulse = impulse * np.exp(-np.linspace(0, 10, length))
        return impulse
    
    def apply_all_augmentations(self, audio, augmentations=None):
        if augmentations is None:
            augmentations = ['noise', 'time_shift', 'pitch_shift', 'volume_change']
        
        augmented = audio.copy()
        
        for aug_type in augmentations:
            if aug_type == 'noise':
                augmented = self.add_noise(augmented)
            elif aug_type == 'time_shift':
                augmented = self.time_shift(augmented)
            elif aug_type == 'pitch_shift':
                n_steps = np.random.uniform(-2, 2)
                augmented = self.pitch_shift(augmented, n_steps)
            elif aug_type == 'volume_change':
                db_change = np.random.uniform(-6, 6)
                augmented = self.change_volume(augmented, db_change)
            elif aug_type == 'time_stretch':
                rate = np.random.uniform(0.8, 1.2)
                augmented = self.time_stretch(augmented, rate)
        
        return augmented
    
    def generate_augmented_batch(self, audio_files, augmentations_per_file=3):
        augmented_files = []
        
        for audio_file in audio_files:
            audio, sr = librosa.load(audio_file, sr=self.sr)
            
            for i in range(augmentations_per_file):
                augmented_audio = self.apply_all_augmentations(audio)
                
                augmented_filename = f"{audio_file}_aug_{i}.wav"
                sf.write(augmented_filename, augmented_audio, sr)
                augmented_files.append(augmented_filename)
        
        return augmented_files