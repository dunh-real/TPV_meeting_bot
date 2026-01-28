import os
from faster_whisper import WhisperModel

def transcribe_audio(input_folder, output_folder, model_size="small"):
    """
    Transcribes audio files using faster-whisper.
    model_size options: "tiny", "base", "small", "medium", "large-v3"
    'small' is the best balance for speed/accuracy on most CPUs.
    """
    
    # Initialize the model
    # Use "cuda" if you have an NVIDIA GPU, otherwise "cpu"
    device = "cpu" 
    print(f"--- Loading model '{model_size}' on {device}... ---")
    model = WhisperModel(model_size, device=device, compute_type="int8")

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Supported extensions
    extensions = ('.mp3', '.wav')

    for filename in os.listdir(input_folder):
        if filename.lower().endswith(extensions):
            file_path = os.path.join(input_folder, filename)
            output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

            print(f"Processing: {filename}...")

            # Transcribe
            # beam_size=5 is standard; increase for accuracy, decrease for speed
            segments, info = model.transcribe(file_path, beam_size=5)

            print(f"Detected language: '{info.language}' with probability {info.language_probability:.2f}")

            with open(output_path, "w", encoding="utf-8") as f:
                for segment in segments:
                    f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
            
            print(f"Saved to: {output_path}")

if __name__ == "__main__":
    # CONFIGURATION
    INPUT_DIR = "../data/raw"  # Folder containing your MP3/WAV
    OUTPUT_DIR = "../data/processed"
    
    # "small" is great for Vietnamese/English. Use "medium" if accuracy is too low.
    MODEL = "small" 

    transcribe_audio(INPUT_DIR, OUTPUT_DIR, MODEL)
    print("\nDone! Check the transcriptions folder.")