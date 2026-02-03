# # import speech_recognition as sr

# # def s2t():
# #     recognizer = sr.Recognizer()
# #     with sr.Microphone() as source:
# #         print("Noi de con vo ei ..")
# #         recognizer.adjust_for_ambient_noise(source)
# #         audio = recognizer.listen(source)
# #     try:
# #         print("Dang xu ly ..")
# #         text = recognizer.recognize_google(audio, language = "vi-VN")
# #         print("Ket qua: " + text)
# #     except sr.UnknownValueError:
# #         print("Bo may chiu")
# #     except sr.RequestError as e:
# #         print(f"Loi tao loi tao: {e}")

# # if __name__ == "__main__":
# #     s2t()

# import os
# import sys
# from pydub import AudioSegment
# import speech_recognition as sr

# # 1. Thêm trực tiếp vào PATH của script đang chạy
# ffmpeg_bin_path = r"C:\ffmpeg\bin" # Đảm bảo đường dẫn này đúng tới folder chứa ffmpeg.exe
# os.environ["PATH"] += os.pathsep + ffmpeg_bin_path

# # 2. Gán đường dẫn cụ thể cho pydub
# AudioSegment.converter = os.path.join(ffmpeg_bin_path, "ffmpeg.exe")
# AudioSegment.ffprobe = os.path.join(ffmpeg_bin_path, "ffprobe.exe")

# # Thử in ra để kiểm tra
# # print(f"Đang tìm FFmpeg tại: {AudioSegment.converter}")

# # # Tiếp tục các lệnh xử lý của bạn
# # src = "../data/raw/1.mp3"
# # dst = "1.wav"
# # sound = AudioSegment.from_mp3(src)
# # sound.export(dst, format = "wav")
# # recognizer = sr.Recognizer()
# # with sr.AudioFile(r"./1.wav") as source:
# #     audio_data = recognizer.record(source)
# #     text = recognizer.recognize_google(audio_data, language = "vi-VN")
# #     print(text)


# src = "../data/raw/1.mp3"
# dst = "1.wav"

# # Đọc file MP3
# sound = AudioSegment.from_mp3(src)

# # Ép định dạng về: 16000Hz, Mono (1 kênh), 16-bit để Google nhận diện tốt nhất
# sound = sound.set_frame_rate(16000).set_channels(1)

# # Xuất file WAV
# sound.export(dst, format="wav")

# print("Đã convert xong file WAV chuẩn.")

# recognizer = sr.Recognizer()
# with sr.AudioFile(dst) as source:
#     # Lọc nhiễu để kết quả chính xác hơn
#     recognizer.adjust_for_ambient_noise(source)
#     audio_data = recognizer.record(source)
#     try:
#         text = recognizer.recognize_google(audio_data, language="vi-VN")
#         print("Kết quả chuyển văn bản:")
#         print(text)
#     except sr.UnknownValueError:
#         print("Google không thể hiểu được âm thanh.")
#     except sr.RequestError as e:
#         print(f"Lỗi kết nối API: {e}")

import os
from pydub import AudioSegment
import speech_recognition as sr
from pydub.utils import make_chunks

# 1. Cấu hình FFmpeg (Bắt buộc)
ffmpeg_bin_path = r"C:\ffmpeg\bin"
os.environ["PATH"] += os.pathsep + ffmpeg_bin_path
AudioSegment.converter = os.path.join(ffmpeg_bin_path, "ffmpeg.exe")
AudioSegment.ffprobe = os.path.join(ffmpeg_bin_path, "ffprobe.exe")

def transcribe_long_audio(file_path):
    # Load file âm thanh
    print("Đang tải file audio...")
    audio = AudioSegment.from_file(file_path)
    
    # Chuẩn hóa âm thanh (Google API thích 16kHz, Mono)
    audio = audio.set_frame_rate(16000).set_channels(1)

    # Chia nhỏ file thành các đoạn 45 giây (để đảm bảo không bị Bad Request)
    chunk_length_ms = 45000 
    chunks = make_chunks(audio, chunk_length_ms)
    
    recognizer = sr.Recognizer()
    full_text = []

    print(f"Đã chia file thành {len(chunks)} đoạn nhỏ. Bắt đầu chuyển văn bản...")

    for i, chunk in enumerate(chunks):
        chunk_name = f"temp_chunk_{i}.wav"
        chunk.export(chunk_name, format="wav")

        with sr.AudioFile(chunk_name) as source:
            audio_listened = recognizer.record(source)
            try:
                # Gửi lên Google
                text = recognizer.recognize_google(audio_listened, language="vi-VN")
                print(f"Đoạn {i+1}: {text[:50]}...")
                full_text.append(text)
            except sr.UnknownValueError:
                print(f"Đoạn {i+1}: Không nhận diện được âm thanh.")
            except sr.RequestError as e:
                print(f"Đoạn {i+1}: Lỗi API; {e}")
        
        # Xóa file tạm ngay sau khi xong để nhẹ máy
        os.remove(chunk_name)

    return " ".join(full_text)

# Chạy thử
src = "../data/raw/1.mp3"
if os.path.exists(src):
    final_result = transcribe_long_audio(src)
    with open("ket_qua_hop.txt", "w", encoding="utf-8") as f:
        f.write(final_result)
    print("\n--- XONG! Kết quả đã lưu vào file ket_qua_hop.txt ---")
else:
    print("Không tìm thấy file nguồn!")