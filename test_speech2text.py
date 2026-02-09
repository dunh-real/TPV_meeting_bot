import datetime
import os
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
import torchaudio
import warnings
import soundfile as sf
import numpy as np


if not hasattr(torchaudio, "list_audio_backends"):
    def _list_audio_backends():
        return ["soundfile", "ffmpeg"] 
    torchaudio.list_audio_backends = _list_audio_backends

# --- SUPPRESS WARNINGS ---
# Suppress the symlink warning
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
# Suppress torchcodec and other deprecation warnings
warnings.filterwarnings("ignore")

# Configuration
AUDIO_FILE = "./data/raw/1.wav"
HF_TOKEN = "" # replace with your hugging face token here
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def format_time(seconds):
    """Converts seconds to HH:MM:SS format."""
    return str(datetime.timedelta(seconds=seconds)).split('.')[0]

def transcribe_and_diarize(audio_file):
    print(f"--- Processing: {audio_file} on {DEVICE} ---")
    
    # -----------------------
    # 1. Transcribe (Faster-Whisper)
    # -----------------------
    print("1. Loading Faster-Whisper...")
    model = WhisperModel("medium", device=DEVICE, compute_type="float16")

    print("   Starting transcription...")
    segments, info = model.transcribe(audio_file, beam_size=5)
    whisper_segments = list(segments)
    print(f"   Transcription done. Language: {info.language}")

    # -----------------------
    # 2. Diarize (Pyannote Manual Load)
    # -----------------------
    print("2. Loading Pyannote Pipeline...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-community-1", 
            token=HF_TOKEN
        ).to(torch.device(DEVICE))
    except Exception as e:
        print(f"Error loading Pyannote: {e}")
        return

    print("   Loading audio manually (via SoundFile)...")
    try:
        # Load using soundfile (bypasses the broken torchcodec)
        waveform_np, sample_rate = sf.read(audio_file)
        
        # Convert Numpy -> Torch Tensor
        # soundfile returns (Time, Channel), but PyTorch needs (Channel, Time)
        if len(waveform_np.shape) == 1:
            # Mono audio: (Time,) -> (1, Time)
            waveform_torch = torch.tensor(waveform_np, dtype=torch.float32).unsqueeze(0)
        else:
            # Stereo/Multi: (Time, Channel) -> (Channel, Time)
            waveform_torch = torch.tensor(waveform_np, dtype=torch.float32).transpose(0, 1)

        audio_in_memory = {"waveform": waveform_torch, "sample_rate": sample_rate}
        
        print("   Starting diarization...")
        diarization_result = pipeline(audio_in_memory)
        
        # Extract timeline
        if hasattr(diarization_result, 'speaker_diarization'):
            diarization = diarization_result.speaker_diarization
        else:
            diarization = diarization_result
            
        print("   Diarization done.")

    except Exception as e:
        print(f"\nDiarization failed: {e}")
        print("NOTE: If this fails with 'libsndfile', install ffmpeg or convert audio to WAV first.")
        return

    # -----------------------
    # 3. Merge Results
    # -----------------------
    print("3. Merging results...")
    final_output = []

    for segment in whisper_segments:
        start_time = segment.start
        end_time = segment.end
        text = segment.text

        speakers_in_segment = []
        
        for turn, _, speaker in diarization.itertracks(yield_label=True):
            intersection_start = max(start_time, turn.start)
            intersection_end = min(end_time, turn.end)
            
            if intersection_end > intersection_start:
                duration = intersection_end - intersection_start
                speakers_in_segment.append((speaker, duration))

        if speakers_in_segment:
            speakers_in_segment.sort(key=lambda x: x[1], reverse=True)
            dominant_speaker = speakers_in_segment[0][0]
        else:
            dominant_speaker = "Unknown"

        final_output.append({
            "start": start_time,
            "end": end_time,
            "speaker": dominant_speaker,
            "text": text.strip()
        })

    # -----------------------
    # 4. Save/Print
    # -----------------------
    print("\n" + "="*50)
    print(f"TRANSCRIPT ({len(final_output)} lines)")
    print("="*50)
    
    with open("transcript.txt", "w", encoding="utf-8") as f:
        for line in final_output:
            time_stamp = f"[{format_time(line['start'])}]"
            formatted_line = f"{time_stamp} {line['speaker']}: {line['text']}"
            print(formatted_line)
            f.write(formatted_line + "\n")

if __name__ == "__main__":
    transcribe_and_diarize(AUDIO_FILE)