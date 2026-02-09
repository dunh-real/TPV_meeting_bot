import subprocess
import os
from dotenv import load_dotenv

load_dotenv()

class AudioConverter:
    def __init__(self) -> None:
        self.ffmpeg = os.getenv("ffmpeg")
        self.output_dir = r"C:\AI Project\Chatbot recording\data_output"
        os.makedirs(self.output_dir, exist_ok=True)

    def convert_audio(self, mp3_input: str):
        # Lấy tên file gốc
        base_name = os.path.splitext(os.path.basename(mp3_input))[0]
        output_wav = os.path.join(self.output_dir, f"{base_name}.wav")  # .wav

        cmd = [
            self.ffmpeg,
            "-y",                     # Overwrite nếu đã tồn tại
            "-i", mp3_input,
            "-ac", "1",
            "-ar", "16000",
            "-f", "wav",              # Buộc format WAV
            "-c:a", "pcm_s16le",
            output_wav
        ]
        print(f"[Converter] Đang chuyển {mp3_input} sang WAV...")
        print(f"[Converter] Lưu tại: {output_wav}")
        try:
            result = subprocess.run(
                cmd,
                check=True,
                capture_output=True,
                encoding='utf-8',      # Buộc dùng UTF-8
                errors='ignore'        # Bỏ qua ký tự không decode được → tránh crash thread
            )
            print(f"[Converter] Chuyển đổi thành công!")
            return output_wav
        except subprocess.CalledProcessError as e:
            print(f"[Converter] Lỗi ffmpeg: {e.stderr}")
            raise RuntimeError(f"Chuyển đổi audio thất bại: {e}")

converter = AudioConverter()