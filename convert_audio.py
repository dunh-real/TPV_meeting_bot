import subprocess

ffmpeg = r"C:\AI Project\Chatbot recording\ffmpeg-8.0.1-essentials_build\bin\ffmpeg.exe"
input_mp3 = r"C:\AI Project\Chatbot recording\data_input\test.mp3"
output_wav = r"C:\AI Project\Chatbot recording\data_output\test.wav"

cmd = [
    ffmpeg,
    "-y",
    "-i", input_mp3,
    "-ac", "1",
    "-ar", "16000",
    "-c:a", "pcm_s16le",
    output_wav
]

subprocess.run(cmd, check=True)
