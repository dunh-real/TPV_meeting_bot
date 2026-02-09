import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torchaudio
import soundfile as sf

class ASRService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.compute_type = "float16" if torch.cuda.is_available() else "int8"
    
    def speech_2_text(self, audio_path):
        return None
    
    def segment_speakers(self, path):
        return None, None