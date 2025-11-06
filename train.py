# train.py
import tensorflow as tf
import numpy as np
import argparse
import yaml
import os

from voiceprint_id.data.audio_processor import AudioProcessor
from voiceprint_id.data.data_augmentation import AudioAugmentation
from voiceprint_id.data.dataset_loader import DatasetLoader

def main():
    parser = argparse.ArgumentParser(description='Train VoicePrint ID Models')
    parser.add_argument('--config', type=str, default='config.yaml', help='Config file path')
    parser.add_argument('--model', type=str, choices=['speaker', 'emotion', 'language', 'spoof'], required=True)
    parser.add_argument('--data_dir', type=str, required=True, help='Dataset directory')
    parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs')
    
    args = parser.parse_args()
    
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)
    
    print(f"Training {args.model} model...")
    
    if args.model == 'speaker':
        train_speaker_model(config, args.data_dir, args.epochs)
    elif args.model == 'emotion':
        train_emotion_model(config, args.data_dir, args.epochs)
    elif args.model == 'language':
        train_language_model(config, args.data_dir, args.epochs)
    elif args.model == 'spoof':
        train_spoof_model(config, args.data_dir, args.epochs)

def train_speaker_model(config, data_dir, epochs):
    from voiceprint_id.models.speaker_models import SpeakerClassifier
    
    loader = DatasetLoader(data_dir)
    audio_data, speaker_labels = loader.load_librispeech_speaker()
    
    feature_utils = FeatureUtils()
    features = feature_utils.create_feature_matrix(audio_data, config['audio']['sample_rate'])
    labels_encoded = feature_utils.encode_labels(speaker_labels)
    
    num_speakers = len(np.unique(labels_encoded))
    model = SpeakerClassifier(num_speakers=num_speakers)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        features, labels_encoded, test_size=config['training']['validation_split'], random_state=42
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=config['training']['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5)
        ]
    )
    
    model.save_weights('models/speaker_classifier.h5')
    print("Speaker model trained and saved.")

def train_emotion_model(config, data_dir, epochs):
    from voiceprint_id.models.emotion_models import EmotionCNN
    
    loader = DatasetLoader(data_dir)
    audio_data, emotion_labels = loader.load_ravdess_emotion()
    
    emotion_labels_encoded = LabelEncoder().fit_transform(emotion_labels)
    num_emotions = len(np.unique(emotion_labels_encoded))
    
    model = EmotionCNN(num_emotions=num_emotions)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    audio_processor = AudioProcessor()
    features = []
    for audio in audio_data:
        mfcc = audio_processor.extract_mfcc(audio, config['audio']['sample_rate'])
        mfcc = mfcc[:, :300]
        if mfcc.shape[1] < 300:
            mfcc = np.pad(mfcc, ((0, 0), (0, 300 - mfcc.shape[1])), mode='constant')
        mfcc = np.expand_dims(mfcc, axis=-1)
        features.append(mfcc)
    
    features = np.array(features)
    
    X_train, X_val, y_train, y_val = train_test_split(
        features, emotion_labels_encoded, test_size=config['training']['validation_split'], random_state=42
    )
    
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=config['training']['batch_size'],
        validation_data=(X_val, y_val),
        callbacks=[
            tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            tf.keras.callbacks.ReduceLROnPlateau(patience=5)
        ]
    )
    
    model.save_weights('models/emotion_classifier.h5')
    print("Emotion model trained and saved.")

if __name__ == "__main__":
    main()