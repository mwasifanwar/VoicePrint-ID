# utils/visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import librosa.display
import plotly.graph_objects as go
from plotly.subplots import make_subplots

class AudioVisualizer:
    def __init__(self):
        plt.style.use('seaborn-v0_8')
        self.colors = plt.rcParams['axes.prop_cycle'].by_key()['color']
    
    def plot_waveform(self, audio, sr, title="Audio Waveform"):
        plt.figure(figsize=(12, 4))
        librosa.display.waveshow(audio, sr=sr, alpha=0.7)
        plt.title(title)
        plt.xlabel('Time (s)')
        plt.ylabel('Amplitude')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_spectrogram(self, audio, sr, title="Spectrogram"):
        plt.figure(figsize=(12, 6))
        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        librosa.display.specshow(D, sr=sr, x_axis='time', y_axis='log')
        
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_mfcc(self, mfccs, sr, title="MFCC Features"):
        plt.figure(figsize=(12, 6))
        
        librosa.display.specshow(mfccs, sr=sr, x_axis='time')
        plt.colorbar()
        plt.title(title)
        plt.ylabel('MFCC Coefficients')
        plt.tight_layout()
        return plt.gcf()
    
    def plot_mel_spectrogram(self, audio, sr, title="Mel Spectrogram"):
        plt.figure(figsize=(12, 6))
        
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        
        librosa.display.specshow(mel_spec_db, sr=sr, x_axis='time', y_axis='mel')
        plt.colorbar(format='%+2.0f dB')
        plt.title(title)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_feature_comparison(self, features1, features2, labels=['Feature Set 1', 'Feature Set 2']):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        ax1.imshow(features1, aspect='auto', origin='lower')
        ax1.set_title(labels[0])
        ax1.set_xlabel('Time Frames')
        ax1.set_ylabel('Features')
        
        ax2.imshow(features2, aspect='auto', origin='lower')
        ax2.set_title(labels[1])
        ax2.set_xlabel('Time Frames')
        ax2.set_ylabel('Features')
        
        plt.tight_layout()
        return fig
    
    def plot_embedding_projections(self, embeddings, labels, title="Speaker Embeddings"):
        from sklearn.manifold import TSNE
        
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_2d = tsne.fit_transform(embeddings)
        
        plt.figure(figsize=(10, 8))
        
        unique_labels = np.unique(labels)
        for i, label in enumerate(unique_labels):
            mask = labels == label
            plt.scatter(embeddings_2d[mask, 0], embeddings_2d[mask, 1], 
                       label=label, alpha=0.7, s=50)
        
        plt.title(title)
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        return plt.gcf()
    
    def plot_confusion_matrix(self, cm, classes, title="Confusion Matrix"):
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=classes, yticklabels=classes)
        
        plt.title(title)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.tight_layout()
        return plt.gcf()
    
    def create_interactive_audio_plot(self, audio, sr, title="Interactive Audio Analysis"):
        times = np.linspace(0, len(audio) / sr, len(audio))
        
        fig = make_subplots(rows=2, cols=2,
                           subplot_titles=('Waveform', 'Spectrogram', 'MFCC', 'Mel Spectrogram'),
                           specs=[[{"secondary_y": False}, {"secondary_y": False}],
                                 [{"secondary_y": False}, {"secondary_y": False}]])
        
        fig.add_trace(go.Scatter(x=times, y=audio, mode='lines', name='Waveform'),
                     row=1, col=1)
        
        D = librosa.amplitude_to_db(np.abs(librosa.stft(audio)), ref=np.max)
        fig.add_trace(go.Heatmap(z=D, colorscale='Viridis'),
                     row=1, col=2)
        
        mfccs = librosa.feature.mfcc(y=audio, sr=sr)
        fig.add_trace(go.Heatmap(z=mfccs, colorscale='Plasma'),
                     row=2, col=1)
        
        mel_spec = librosa.feature.melspectrogram(y=audio, sr=sr)
        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)
        fig.add_trace(go.Heatmap(z=mel_spec_db, colorscale='Hot'),
                     row=2, col=2)
        
        fig.update_layout(height=800, title_text=title)
        return fig
    
    def plot_training_history(self, history):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        ax1.plot(history.history['accuracy'], label='Training Accuracy')
        ax1.plot(history.history['val_accuracy'], label='Validation Accuracy')
        ax1.set_title('Model Accuracy')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        ax2.plot(history.history['loss'], label='Training Loss')
        ax2.plot(history.history['val_loss'], label='Validation Loss')
        ax2.set_title('Model Loss')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig