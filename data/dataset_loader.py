# data/dataset_loader.py
import os
import numpy as np
import pandas as pd
import librosa
from sklearn.model_selection import train_test_split

class DatasetLoader:
    def __init__(self, data_dir, target_sr=16000):
        self.data_dir = data_dir
        self.target_sr = target_sr
        self.datasets = {}
    
    def load_ravdess_emotion(self):
        emotion_labels = {
            '01': 'neutral', '02': 'calm', '03': 'happy', '04': 'sad',
            '05': 'angry', '06': 'fearful', '07': 'disgust', '08': 'surprised'
        }
        
        data = []
        labels = []
        
        for root, dirs, files in os.walk(self.data_dir):
            for file in files:
                if file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    
                    parts = file.split('-')
                    if len(parts) >= 3:
                        emotion_code = parts[2]
                        emotion = emotion_labels.get(emotion_code, 'unknown')
                        
                        audio, sr = librosa.load(file_path, sr=self.target_sr)
                        
                        data.append(audio)
                        labels.append(emotion)
        
        return data, labels
    
    def load_librispeech_speaker(self, max_speakers=100):
        data = []
        speaker_ids = []
        
        speaker_count = 0
        for root, dirs, files in os.walk(self.data_dir):
            if speaker_count >= max_speakers:
                break
                
            for file in files:
                if file.endswith('.flac') or file.endswith('.wav'):
                    file_path = os.path.join(root, file)
                    
                    speaker_id = os.path.basename(root)
                    
                    audio, sr = librosa.load(file_path, sr=self.target_sr, duration=5.0)
                    
                    data.append(audio)
                    speaker_ids.append(speaker_id)
            
            speaker_count += 1
        
        return data, speaker_ids
    
    def load_common_voice_language(self):
        data = []
        languages = []
        
        metadata_path = os.path.join(self.data_dir, 'validated.tsv')
        
        if os.path.exists(metadata_path):
            df = pd.read_csv(metadata_path, sep='\t')
            
            for _, row in df.iterrows():
                audio_path = os.path.join(self.data_dir, 'clips', row['path'])
                language = row['locale']
                
                if os.path.exists(audio_path):
                    try:
                        audio, sr = librosa.load(audio_path, sr=self.target_sr, duration=5.0)
                        data.append(audio)
                        languages.append(language)
                    except:
                        continue
        
        return data, languages
    
    def create_speaker_verification_pairs(self, audio_files, speaker_labels, num_pairs=1000):
        positive_pairs = []
        negative_pairs = []
        
        speaker_audio_map = {}
        for audio, speaker in zip(audio_files, speaker_labels):
            if speaker not in speaker_audio_map:
                speaker_audio_map[speaker] = []
            speaker_audio_map[speaker].append(audio)
        
        speakers = list(speaker_audio_map.keys())
        
        for _ in range(num_pairs // 2):
            speaker = np.random.choice(speakers)
            if len(speaker_audio_map[speaker]) >= 2:
                audio1, audio2 = np.random.choice(speaker_audio_map[speaker], 2, replace=False)
                positive_pairs.append((audio1, audio2, 1))
        
        for _ in range(num_pairs // 2):
            speaker1, speaker2 = np.random.choice(speakers, 2, replace=False)
            audio1 = np.random.choice(speaker_audio_map[speaker1])
            audio2 = np.random.choice(speaker_audio_map[speaker2])
            negative_pairs.append((audio1, audio2, 0))
        
        all_pairs = positive_pairs + negative_pairs
        np.random.shuffle(all_pairs)
        
        return all_pairs
    
    def split_dataset(self, data, labels, test_size=0.2, val_size=0.1):
        X_temp, X_test, y_temp, y_test = train_test_split(
            data, labels, test_size=test_size, random_state=42, stratify=labels
        )
        
        val_ratio = val_size / (1 - test_size)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=val_ratio, random_state=42, stratify=y_temp
        )
        
        return (X_train, y_train), (X_val, y_val), (X_test, y_test)
    
    def extract_features_batch(self, audio_list, feature_type='mfcc', **kwargs):
        features = []
        
        for audio in audio_list:
            if feature_type == 'mfcc':
                feature = librosa.feature.mfcc(y=audio, sr=self.target_sr, **kwargs)
            elif feature_type == 'mel_spectrogram':
                feature = librosa.feature.melspectrogram(y=audio, sr=self.target_sr, **kwargs)
                feature = librosa.power_to_db(feature)
            elif feature_type == 'chroma':
                feature = librosa.feature.chroma_stft(y=audio, sr=self.target_sr, **kwargs)
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
            
            features.append(feature)
        
        return features