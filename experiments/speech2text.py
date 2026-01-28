# import os
# from faster_whisper import WhisperModel
# import time
# import torch

# def transcribe_audio(input_folder, output_folder, model_size="small"):
#     """
#     Transcribes audio files using faster-whisper.
#     model_size options: "tiny", "base", "small", "medium", "large-v3"
#     'small' is the best balance for speed/accuracy on most CPUs.
#     """
    
#     # Initialize the model
#     # Use "cuda" if you have an NVIDIA GPU, otherwise "cpu"
#     device = "cpu" 
#     print(f"--- Loading model '{model_size}' on {device}... ---")
#     model = WhisperModel(model_size, device=device, compute_type="int8")

#     if not os.path.exists(output_folder):
#         os.makedirs(output_folder)

#     # Supported extensions
#     extensions = ('.mp3', '.wav')

#     for filename in os.listdir(input_folder):
#         if filename.lower().endswith(extensions):
#             file_path = os.path.join(input_folder, filename)
#             output_path = os.path.join(output_folder, f"{os.path.splitext(filename)[0]}.txt")

#             print(f"Processing: {filename}...")

#             # Transcribe
#             # beam_size=5 is standard; increase for accuracy, decrease for speed
#             segments, info = model.transcribe(file_path, beam_size=5)

#             print(f"Detected language: '{info.language}' with probability {info.language_probability:.2f}")

#             with open(output_path, "w", encoding="utf-8") as f:
#                 for segment in segments:
#                     f.write(f"[{segment.start:.2f}s -> {segment.end:.2f}s] {segment.text}\n")
            
#             print(f"Saved to: {output_path}")

# if __name__ == "__main__":
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     start_time = time.perf_counter()

#     # CONFIGURATION
#     INPUT_DIR = "../data/raw"  # Folder containing your MP3/WAV
#     OUTPUT_DIR = "../data/processed"
    
#     # "small" is great for Vietnamese/English. Use "medium" if accuracy is too low.
#     MODEL = "medium"
#     MODEL = MODEL.to(device)

#     transcribe_audio(INPUT_DIR, OUTPUT_DIR, MODEL)

#     end_time = time.perf_counter()
#     elapsed_time = end_time - start_time
#     print("\nDone! Check the transcriptions folder.")
#     print(f"\nElapsed time: {elapsed_time:.4f} seconds")

import nemo.collections.asr as nemo_asr

asr_model = nemo_asr.models.ASRModel.from_pretrained(model_name = "nvidia/parakeet-ctc-0.6b-Vietnamese")

output = asr_model.transcribe(['../data/raw/1.mp3'])
print(output[0].text)