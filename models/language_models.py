# models/language_models.py
import tensorflow as tf

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