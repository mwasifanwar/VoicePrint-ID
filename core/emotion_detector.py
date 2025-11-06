# core/emotion_detector.py
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf

class EmotionCNN(tf.keras.Model):
    def __init__(self, num_emotions=7):
        super(EmotionCNN, self).__init__()
        
        self.conv_layers = tf.keras.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.MaxPooling2D((2, 2)),
            tf.keras.layers.Dropout(0.25),
            
            tf.keras.layers.Conv2D(256, (3, 3), activation='relu', padding='same'),
            tf.keras.layers.BatchNormalization(),
            tf.keras.layers.GlobalAveragePooling2D()
        ])
        
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.5),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_emotions, activation='softmax')
        ])
    
    def call(self, inputs, training=False):
        x = self.conv_layers(inputs, training=training)
        return self.classifier(x, training=training)

class EmotionDetector:
    def __init__(self, model_path=None):
        self.model = EmotionCNN()
        self.sr = 16000
        self.n_mfcc = 40
        
        self.emotions = {
            0: 'neutral',
            1: 'happy',
            2: 'sad',
            3: 'angry',
            4: 'fearful',
            5: 'disgust',
            6: 'surprised'
        }
        
        if model_path:
            self.load_model(model_path)
    
    def extract_emotion_features(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr)
            
            if len(audio) < int(self.sr * 1.0):
                audio = np.pad(audio, (0, max(0, int(self.sr * 1.0) - len(audio))))
            
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=2048, hop_length=512)
            mfccs = mfccs[:, :300]
            
            if mfccs.shape[1] < 300:
                pad_width = 300 - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            
            mfccs = np.expand_dims(mfccs, axis=-1)
            mfccs = np.expand_dims(mfccs, axis=0)
            
            return mfccs
            
        except Exception as e:
            print(f"Emotion feature extraction error: {e}")
            return None
    
    def detect_emotion(self, audio_path):
        features = self.extract_emotion_features(audio_path)
        if features is None:
            return None, 0.0
        
        predictions = self.model(features, training=False)
        emotion_idx = np.argmax(predictions[0])
        confidence = np.max(predictions[0])
        
        emotion = self.emotions.get(emotion_idx, 'unknown')
        
        return emotion, float(confidence)
    
    def detect_emotion_from_audio(self, audio_array, sr):
        try:
            if sr != self.sr:
                audio_array = librosa.resample(audio_array, orig_sr=sr, target_sr=self.sr)
            
            mfccs = librosa.feature.mfcc(y=audio_array, sr=self.sr, n_mfcc=self.n_mfcc)
            mfccs = mfccs[:, :300]
            
            if mfccs.shape[1] < 300:
                pad_width = 300 - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            
            mfccs = np.expand_dims(mfccs, axis=-1)
            mfccs = np.expand_dims(mfccs, axis=0)
            
            predictions = self.model(mfccs, training=False)
            emotion_idx = np.argmax(predictions[0])
            confidence = np.max(predictions[0])
            
            emotion = self.emotions.get(emotion_idx, 'unknown')
            
            return emotion, float(confidence)
            
        except Exception as e:
            print(f"Real-time emotion detection error: {e}")
            return None, 0.0
    
    def save_model(self, model_path):
        self.model.save_weights(model_path)
    
    def load_model(self, model_path):
        self.model.build(input_shape=(None, 40, 300, 1))
        self.model.load_weights(model_path)