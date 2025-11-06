# core/anti_spoofing.py
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf

class AntiSpoofingModel(tf.keras.Model):
    def __init__(self):
        super(AntiSpoofingModel, self).__init__()
        
        self.conv_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(2, activation='softmax')
        ])
    
    def call(self, inputs, training=False):
        x = self.conv_layers(inputs, training=training)
        return self.classifier(x, training=training)

class AntiSpoofing:
    def __init__(self, model_path=None):
        self.model = AntiSpoofingModel()
        self.sr = 16000
        self.n_mfcc = 40
        
        if model_path:
            self.load_model(model_path)
    
    def extract_spoof_features(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, duration=3.0)
            
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=2048, hop_length=512)
            mfccs = mfccs[:, :300]
            
            if mfccs.shape[1] < 300:
                pad_width = 300 - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            
            spectral_centroid = librosa.feature.spectral_centroid(y=audio, sr=sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio, sr=sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio)
            
            additional_features = np.vstack([spectral_centroid, spectral_rolloff, zero_crossing_rate])
            additional_features = additional_features[:, :300]
            
            if additional_features.shape[1] < 300:
                pad_width = 300 - additional_features.shape[1]
                additional_features = np.pad(additional_features, ((0, 0), (0, pad_width)), mode='constant')
            
            combined_features = np.vstack([mfccs, additional_features])
            combined_features = np.expand_dims(combined_features, axis=-1)
            combined_features = np.expand_dims(combined_features, axis=0)
            
            return combined_features
            
        except Exception as e:
            print(f"Spoof feature extraction error: {e}")
            return None
    
    def detect_spoof(self, audio_path):
        features = self.extract_spoof_features(audio_path)
        if features is None:
            return False, 0.0
        
        predictions = self.model(features, training=False)
        is_real = np.argmax(predictions[0]) == 1
        confidence = predictions[0][1] if is_real else predictions[0][0]
        
        return bool(is_real), float(confidence)
    
    def detect_spoof_from_audio(self, audio_array, sr):
        try:
            if sr != self.sr:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.sr)
            
            mfccs = librosa.feature.mfcc(y=audio_array, sr=self.sr, n_mfcc=self.n_mfcc)
            mfccs = mfccs[:, :300]
            
            if mfccs.shape[1] < 300:
                pad_width = 300 - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            
            spectral_centroid = librosa.feature.spectral_centroid(y=audio_array, sr=self.sr)
            spectral_rolloff = librosa.feature.spectral_rolloff(y=audio_array, sr=self.sr)
            zero_crossing_rate = librosa.feature.zero_crossing_rate(y=audio_array)
            
            additional_features = np.vstack([spectral_centroid, spectral_rolloff, zero_crossing_rate])
            additional_features = additional_features[:, :300]
            
            if additional_features.shape[1] < 300:
                pad_width = 300 - additional_features.shape[1]
                additional_features = np.pad(additional_features, ((0, 0), (0, pad_width)), mode='constant')
            
            combined_features = np.vstack([mfccs, additional_features])
            combined_features = np.expand_dims(combined_features, axis=-1)
            combined_features = np.expand_dims(combined_features, axis=0)
            
            predictions = self.model(combined_features, training=False)
            is_real = np.argmax(predictions[0]) == 1
            confidence = predictions[0][1] if is_real else predictions[0][0]
            
            return bool(is_real), float(confidence)
            
        except Exception as e:
            print(f"Real-time spoof detection error: {e}")
            return False, 0.0
    
    def save_model(self, model_path):
        self.model.save_weights(model_path)
    
    def load_model(self, model_path):
        self.model.build(input_shape=(None, 43, 300, 1))
        self.model.load_weights(model_path)