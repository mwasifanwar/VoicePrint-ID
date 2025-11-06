# __init__.py
__version__ = "1.0.0"
__author__ = "mwasifanwar"
__description__ = "VoicePrint ID: Multi-Speaker Recognition System"

from voiceprint_id.core.speaker_recognizer import SpeakerRecognizer
from voiceprint_id.core.emotion_detector import EmotionDetector
from voiceprint_id.core.language_detector import LanguageDetector
from voiceprint_id.core.anti_spoofing import AntiSpoofing
from voiceprint_id.core.voice_enhancer import VoiceEnhancer