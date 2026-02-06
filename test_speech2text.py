import datetime
import torch
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline

# Configuration
AUDIO_FILE = "./data/raw/1.mp3"
HF_TOKEN = ""
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def milisec(timeStr):
    """Converts seconds to miliseconds"""
    spl = timeStr.split(":")
    s = (int)((int(spl[0]) * 60 * 60 + int(spl[1]) *60 + float(spl[2])) * 1000)
    return s

def format_time(seconds):
    """Converts seconds to HH:MM:SS format"""
    return str(datetime.timedelta(seconds = seconds)).split('.')[0]

def transcribe_and_diarize(audio_file):
    print(f"--- Processing: {audio_file} on {DEVICE} ---")
    print("1. Loading Faster-Whisper model...")
    # use 'medium' for balance
    model_size = 'medium'
    
    # compute type: "float16" for GPU, "int8" for CPU
    compute_type = "float16" if DEVICE == "cuda" else "int8"
    
    model = WhisperModel(
        model_size,
        device = DEVICE,
        compute_type = compute_type
    )
    print("  Starting transcription...")
    segments, info = model.transcribe(audio_file, beam_size = 5)
    
    # convert generator to list to reuse it later
    whisper_segments = list(segments)
    print(f"  Transcription done. Detected language: {info.language}")
    
    print("2. Loading Pyannote Diarization pipeline...")
    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token = HF_TOKEN
        ).to(torch.device(DEVICE))
    except Exception as e:
        print(f"Error loading Pyannote: {e}")
        print("Ensure you accepted the user agreement on HuggingFace for 'speaker-diarization-3.1' and 'segmentation-3.0'")
        return
    
    print("  Starting diarization...")
    diarization = pipeline(audio_file)
    print("  Diarization done.")
    
    # Merge transcription and diarization
    print("3. Merging results...")
    final_output = []
    
    # iterate over whisper segments
    for segment in whisper_segments:
        start_time = segment.start
        end_time = segment.end
        text = segment.text
        
        # find the speaker who spoke the most during this segment
        speakers_in_segment = []
        
        # iterate over diarization timestamps to find overlap
        for turn, _, speaker in diarization.itertracks(yield_label = True):
            # check for overlap between whisper segment and diarization turn
            intersection_start = max(start_time, turn.start)
            intersection_end = min(end_time, turn.end)
            
            if intersection_end > intersection_start:
                duration = intersection_end - intersection_start
                speakers_in_segment.append((speaker, duration))
        
        # determine dominant speaker
        if speakers_in_segment:
            # sort by duration descending and pick the top one
            speakers_in_segment.sort(key = lambda x: x[1], reverse = True)
            dominant_speaker = speakers_in_segment[0][0]
        else:
            dominant_speaker = "Unknown"
        
        final_output.append({
            "start": start_time,
            "end": end_time,
            "speaker": dominant_speaker,
            "text": text.strip()
        })
    
    print("\n" + "="*70)
    print("FINAL TRANSCRIPT")
    print("="*70)
    
    with open("transcript.txt", 'w', encoding = 'utf-8') as f:
        for line in final_output:
            time_stamp = f"[{format_time(line['start'])} --> {format_time(line['end'])}]"