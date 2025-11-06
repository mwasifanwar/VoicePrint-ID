# core/speaker_recognizer.py
import tensorflow as tf
import numpy as np
import librosa
import soundfile as sf
from sklearn.preprocessing import StandardScaler
import pickle
import os

class SpeakerEncoder(tf.keras.Model):
    def __init__(self, embedding_dim=256):
        super(SpeakerEncoder, self).__init__()
        
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
        
        self.attention = tf.keras.layers.MultiHeadAttention(num_heads=8, key_dim=64)
        
        self.dense_layers = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.2),
            tf.keras.layers.Dense(embedding_dim, activation='linear', name='embedding')
        ])
    
    def call(self, inputs, training=False):
        x = self.conv_layers(inputs, training=training)
        
        seq_len = tf.shape(x)[1]
        attention_input = tf.reshape(x, (-1, seq_len, 256))
        
        attended = self.attention(attention_input, attention_input)
        attended_pooled = tf.reduce_mean(attended, axis=1)
        
        embedding = self.dense_layers(attended_pooled, training=training)
        embedding = tf.nn.l2_normalize(embedding, axis=1)
        
        return embedding

class SpeakerRecognizer:
    def __init__(self, model_path=None):
        self.encoder = SpeakerEncoder()
        self.scaler = StandardScaler()
        self.speaker_database = {}
        self.embedding_dim = 256
        
        if model_path and os.path.exists(model_path):
            self.load_model(model_path)
        
        self.sr = 16000
        self.n_mfcc = 40
        self.duration = 3.0
        
    def extract_features(self, audio_path):
        try:
            audio, sr = librosa.load(audio_path, sr=self.sr, duration=self.duration)
            
            if len(audio) < int(self.sr * 0.5):
                raise ValueError("Audio too short")
            
            mfccs = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=self.n_mfcc, n_fft=2048, hop_length=512)
            mfccs = mfccs[:, :300]
            
            if mfccs.shape[1] < 300:
                pad_width = 300 - mfccs.shape[1]
                mfccs = np.pad(mfccs, ((0, 0), (0, pad_width)), mode='constant')
            
            mfccs = np.expand_dims(mfccs, axis=-1)
            mfccs = np.expand_dims(mfccs, axis=0)
            
            return mfccs
            
        except Exception as e:
            print(f"Feature extraction error: {e}")
            return None
    
    def create_embedding(self, audio_path):
        features = self.extract_features(audio_path)
        if features is None:
            return None
        
        embedding = self.encoder(features, training=False)
        return embedding.numpy()[0]
    
    def register_speaker(self, speaker_id, audio_paths):
        embeddings = []
        
        for audio_path in audio_paths:
            embedding = self.create_embedding(audio_path)
            if embedding is not None:
                embeddings.append(embedding)
        
        if len(embeddings) == 0:
            return False
        
        avg_embedding = np.mean(embeddings, axis=0)
        self.speaker_database[speaker_id] = avg_embedding
        
        return True
    
    def identify_speaker(self, audio_path, threshold=0.7):
        embedding = self.create_embedding(audio_path)
        if embedding is None:
            return None, 0.0
        
        best_match = None
        best_score = 0.0
        
        for speaker_id, registered_embedding in self.speaker_database.items():
            similarity = np.dot(embedding, registered_embedding)
            
            if similarity > best_score and similarity > threshold:
                best_score = similarity
                best_match = speaker_id
        
        return best_match, best_score
    
    def verify_speaker(self, audio_path, claimed_speaker_id, threshold=0.7):
        embedding = self.create_embedding(audio_path)
        if embedding is None:
            return False, 0.0
        
        if claimed_speaker_id not in self.speaker_database:
            return False, 0.0
        
        registered_embedding = self.speaker_database[claimed_speaker_id]
        similarity = np.dot(embedding, registered_embedding)
        
        return similarity >= threshold, similarity
    
    def save_model(self, model_path):
        self.encoder.save_weights(model_path)
        
        database_path = model_path.replace('.h5', '_database.pkl')
        with open(database_path, 'wb') as f:
            pickle.dump(self.speaker_database, f)
    
    def load_model(self, model_path):
        self.encoder.build(input_shape=(None, 40, 300, 1))
        self.encoder.load_weights(model_path)
        
        database_path = model_path.replace('.h5', '_database.pkl')
        if os.path.exists(database_path):
            with open(database_path, 'rb') as f:
                self.speaker_database = pickle.load(f)