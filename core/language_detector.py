# core/language_detector.py
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf

class LanguageDetectorModel(tf.keras.Model):
    def __init__(self, num_languages=10):
        super(LanguageDetectorModel, self).__init__()
        
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
        
        self.lstm_layers = tf.keras.Sequential([
            tf.keras.layers.Reshape((-1, 256)),
            tf.keras.layers.LSTM(128, return_sequences=True),
            tf.keras.layers.LSTM(64, return_sequences=False),
        ])
        
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(128, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_languages, activation='softmax')
        ])
    
    def call(self, inputs, training=False):
        x = self.conv_layers(inputs, training=training)
        x = self.lstm_layers(x, training=training)
        return self.classifier(x, training=training)

class LanguageDetector:
    def __init__(self, model_path=None):
        self.model = LanguageDetectorModel()
        self.sr = 16000
        self.n_mfcc = 40
        
        self.languages = {
            0: 'english',
            1: 'spanish', 
            2: 'french',
            3: 'german',
            4: 'italian',
            5: 'mandarin',
            6: 'hindi',
            7: 'arabic',
            8: 'japanese',
            9: 'russian'
        }
        
        if model_path:
            self.load_model(model_path)
    
    def extract_language_features(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, duration=5.0)
            
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=2048, hop_length=512)
            mfccs = mfccs[:, :400]
            
            if mfccs.shape[1] < 400:
                pad_width = 400 - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            
            mfccs = np.expand_dims(mfccs, axis=-1)
            mfccs = np.expand_dims(mfccs, axis=0)
            
            return mfccs
            
        except Exception as e:
            print(f"Language feature extraction error: {e}")
            return None
    
    def detect_language(self, audio_path):
        features = self.extract_language_features(audio_path)
        if features is None:
            return None, 0.0
        
        predictions = self.model(features, training=False)
        language_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        language = self.languages.get(language_idx, 'unknown')
        
        return language, float(confidence)
    
    def detect_language_from_audio(self, audio_array, sr):
        try:
            if sr != self.sr:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.sr)
            
            if len(audio_array) < self.sr * 2:
                audio_array = np.pad(audio_array, (0, max(0, int(self.sr * 2) - len(audio_array))))
            
            mfccs = librosa.feature.mfcc(y=audio_array, sr=self.sr, n_mfcc=self.n_mfcc)
            mfccs = mfccs[:, :400]
            
            if mfccs.shape[1] < 400:
                pad_width = 400 - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            
            mfccs = np.expand_dims(mfccs, axis=-1)
            mfccs = np.expand_dims(mfccs, axis=0)
            
            predictions = self.model(mfccs, training=False)
            language_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            language = self.languages.get(language_idx, 'unknown')
            
            return language, float(confidence)
            
        except Exception as e:
            print(f"Real-time language detection error: {e}")
            return None, 0.0
    
    def save_model(self, model_path):
        self.model.save_weights(model_path)
    
    def load_model(self, model_path):
        self.model.build(input_shape=(None, 40, 400, 1))
        self.model.load_weights(model_path)