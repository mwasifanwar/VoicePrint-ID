# models/speaker_models.py
import tensorflow as tf

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

class SpeakerClassifier(tf.keras.Model):
    def __init__(self, num_speakers=100, embedding_dim=256):
        super(SpeakerClassifier, self).__init__()
        
        self.encoder = SpeakerEncoder(embedding_dim)
        
        self.classifier = tf.keras.Sequential([
            tf.keras.layers.Dense(512, activation='relu'),
            tf.keras.layers.Dropout(0.4),
            tf.keras.layers.Dense(256, activation='relu'),
            tf.keras.layers.Dropout(0.3),
            tf.keras.layers.Dense(num_speakers, activation='softmax')
        ])
    
    def call(self, inputs, training=False):
        embeddings = self.encoder(inputs, training=training)
        return self.classifier(embeddings, training=training)