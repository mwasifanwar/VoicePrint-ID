# utils/feature_utils.py
import numpy as np
import librosa
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA

class FeatureUtils:
    def __init__(self):
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = None
    
    def extract_comprehensive_features(self, audio, sr, n_mfcc=40):
        features = {}
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
        features['mfcc_mean'] = np.mean(mfccs, axis=1)
        features['mfcc_std'] = np.std(mfccs, axis=1)
        
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec)
        features['mel_mean'] = np.mean(mel_spec_db, axis=1)
        features['mel_std'] = np.std(mel_spec_db, axis=1)
        
        chroma = librosa.feature.chroma_stft(y=audio, sr=sr)
        features['chroma_mean'] = np.mean(chroma, axis=1)
        
        spectral_contrast = librosa.feature.spectral_contrast(y=audio, sr=sr)
        features['spectral_contrast_mean'] = np.mean(spectral_contrast, axis=1)
        
        tonnetz = librosa.feature.tonnetz(y=audio, sr=sr)
        features['tonnetz_mean'] = np.mean(tonnetz, axis=1)
        
        features['zero_crossing_rate'] = np.mean(librosa.feature.zero_crossing_rate(audio))
        features['spectral_centroid'] = np.mean(librosa.feature.spectral_centroid(y=audio, sr=sr))
        features['spectral_rolloff'] = np.mean(librosa.feature.spectral_rolloff(y=audio, sr=sr))
        features['rms_energy'] = np.mean(librosa.feature.rms(y=audio))
        
        return np.concatenate([v.flatten() for v in features.values()])
    
    def normalize_features(self, features):
        if not hasattr(self.scaler, 'mean_'):
            self.scaler.fit(features)
        return self.scaler.transform(features)
    
    def encode_labels(self, labels):
        return self.label_encoder.fit_transform(labels)
    
    def decode_labels(self, encoded_labels):
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def apply_pca(self, features, n_components=50):
        if self.pca is None:
            self.pca = PCA(n_components=n_components)
            return self.pca.fit_transform(features)
        else:
            return self.pca.transform(features)
    
    def create_sequence_features(self, audio, sr, sequence_length=100, feature_dim=40):
        mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=feature_dim)
        
        if mfccs.shape[1] < sequence_length:
            pad_width = sequence_length - mfccs.shape[1]
            mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
        else:
            mfccs = mfccs[:, :sequence_length]
        
        return mfccs.T
    
    def extract_delta_features(self, features, order=2):
        delta_features = [features]
        
        for i in range(1, order + 1):
            delta = librosa.feature.delta(features, order=i)
            delta_features.append(delta)
        
        return np.vstack(delta_features)
    
    def create_feature_matrix(self, audio_list, sr, feature_type='comprehensive'):
        feature_vectors = []
        
        for audio in audio_list:
            if feature_type == 'comprehensive':
                features = self.extract_comprehensive_features(audio, sr)
            elif feature_type == 'mfcc':
                mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40)
                features = np.concatenate([np.mean(mfccs, axis=1), np.std(mfccs, axis=1)])
            else:
                raise ValueError(f"Unsupported feature type: {feature_type}")
            
            feature_vectors.append(features)
        
        return np.array(feature_vectors)