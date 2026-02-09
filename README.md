
# Chatbot Meeting Recording

Dự án này chuyển audio cuộc họp (MP3) thành transcript có phân đoạn người nói (speaker diarization), sau đó cho phép hỏi đáp nội dung transcript bằng LLM (đang test với Gemini thông qua LangChain).

**Luồng xử lý chính**
1. MP3 → WAV 16kHz mono bằng FFmpeg.
2. Speaker diarization bằng `pyannote/speaker-diarization-3.1`.
3. ASR tiếng Việt bằng NeMo Parakeet (`parakeet-ctc-0.6b-vi`).
4. Ghi transcript theo format `[start - end] SPEAKER_xx: text`.
5. Làm sạch transcript và chat hỏi đáp bằng Gemini.

**Tính năng**
- Tự động phân đoạn người nói.
- Nhận dạng tiếng Việt với Parakeet CTC.
- Xuất transcript dạng dễ đọc và có timestamp.
- Chat hỏi đáp theo nội dung transcript.

**Yêu cầu**
- Windows + Python 3.12.
- GPU NVIDIA (khuyến nghị).
- FFmpeg (đã có sẵn trong thư mục `ffmpeg-8.0.1-essentials_build`) hoặc tải và làm theo hướng dẫn trong .\ffmpeg-8.0.1-essentials_build\requirements.txt.
- Hugging Face token để tải model pyannote.
- Google Gemini API key để chat.

**Cấu trúc thư mục chính**
- `data_input` chứa file MP3 đầu vào.
- `data_output` chứa WAV tạm (tự tạo).
- `Text` chứa transcript đầu ra.
- `models` chứa model Parakeet `.nemo`.
- `model_detect` chứa model pyannote diarization (hoặc sẽ tự tải).

## Cài đặt

1. Tạo môi trường ảo và cài dependencies

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Ghi chú:
- `torch` và `torchaudio` đang dùng phiên bản CUDA `cu128`. Nếu máy không có CUDA phù hợp, hãy cài bản CUDA tương ứng. Ví dụ cài bản CUDA 12.8:

```powershell
pip install torch==2.8.0+cu128 torchaudio==2.8.0+cu128 --index-url https://download.pytorch.org/whl/cu128
```

2. Chuẩn bị `.env`

Tạo file `.env` theo mẫu sau (chỉnh đường dẫn cho đúng máy):

```env
ffmpeg=".../ffmpeg-8.0.1-essentials_build/bin/ffmpeg.exe"
PARAKET_MODEL_PATH=.../models/parakeet-ctc-0.6b-vi.nemo"
MODEL_DETECTION_PATH=".../model_detect"
HF_TOKEN="hf_xxxYourHuggingFaceTokenxxx"
OUTPUT_FILE=".../Text/transcription.txt"
API_KEY="AIzaSy...YourGeminiAPIKey..."
```

Lưu ý:
- Biến môi trường cho Hugging Face **phải là `HF_TOKEN`** (đúng như `down_diarization.py`).
- Không commit `.env` lên repo.

3. Tải model Parakeet

Chạy:

```powershell
python download_parakeet.py
```

Nếu bạn đặt project ở đường dẫn khác, hãy sửa `local_dir` trong `download_parakeet.py` hoặc sửa `PARAKET_MODEL_PATH` trong `.env` cho đúng.

4. Tải model diarization

`extract_text.py` tự gọi `ensure_diarization_model()` và sẽ tải model nếu chưa có. Để tải trước:

```powershell
python down_diarization.py
```

Bạn cần đăng nhập và chấp nhận điều khoản các model sau trên Hugging Face:

Dưới đây là danh sách 3 model bắt buộc bạn cần phải vào bấm "Agree and access repository" để hệ thống Pyannote 3.1 hoạt động được (bạn cần đăng nhập cùng một tài khoản Hugging Face mà bạn đã lấy Token):

Model chính (Diarization): 👉 https://huggingface.co/pyannote/speaker-diarization-3.1

Model phân đoạn (Segmentation - Quan trọng): 👉 https://huggingface.co/pyannote/segmentation-3.0

Model bổ trợ (Community/Legacy): 👉 https://huggingface.co/pyannote/speaker-diarization-community-1

## Sử dụng

### 1) Tạo transcript có diarization

```powershell
python extract_text.py
```

Nhập đường dẫn file MP3 khi được hỏi. Kết quả sẽ ghi ra file ở `OUTPUT_FILE` trong `.env`.

Định dạng output:
```
[12.34s - 18.90s] SPEAKER_00: nội dung nhận dạng
```

### 2) Làm sạch transcript (gộp các đoạn cùng speaker liên tiếp)

Có thể dùng nhanh bằng cách sửa biến `file_path` trong `clean_text.py`, sau đó chạy:

```powershell
python clean_text.py
```

Hoặc gọi trong Python:

```python
from clean_text import TextCleaner
TextCleaner.clean_and_overwrite(".../path/to/transcript.txt")
```

### 3) Chat hỏi đáp theo transcript

```powershell
python llm_extracter.py
```

Làm theo hướng dẫn để nhập file transcript. Các lệnh trong chat:
- `new` để đổi file transcript.
- `exit` hoặc `quit` để thoát.

## Ghi chú kỹ thuật

- Hiện tại đang xử lý cho video > 2 tiếng với tốc độ < 7 phút (còn tùy. vào lượng âm thanh trong video)
- `extract_text.py` sẽ xóa file WAV tạm sau khi xử lý xong.
- Tốc độ và chất lượng phụ thuộc GPU. CPU vẫn chạy nhưng chậm hơn nhiều.
- Nếu gặp lỗi load `pyannote` do không tương thích phiên bản, hãy thử align về `pyannote-audio==3.1.1` (file `model_detect/requirements.txt`) hoặc nâng cấp đồng bộ các package liên quan.

## File chính

- `download_parakeet.py` tải model parakeet vào `PARAKET_MODEL_PATH`.
- `down_diarization.py` tải model pyannote vào `MODEL_DETECTION_PATH`.
- `load_parakeet.py` load model Parakeet.
- `load_diarization.py` load model `pyannote/speaker-diarization-3.1`.
- `convert_audio.py` chuyển MP3 → WAV bằng FFmpeg.
- `extract_text.py` pipeline chuyển MP3 → diarization → ASR → transcript.
- `clean_text.py` gộp transcript để dễ đọc.
- `llm_service.py` khởi tạo model LLM.
- `llm_extracter.py` chat với transcript bằng Gemini.

Cài đặt đầy đủ theo hướng dẫn rồi tạo 2 terminal: terminal 1 chạy file `extract_text.py` để lấy file transcription.txt, terminal 2 chạy  `llm_extracter.py` (nhập đường dẫn từ file transcript.txt) để nạp dữ liệu cho LLM để hoàn thành pipeline end-to-end.
