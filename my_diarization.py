import torch
import gc
import os
import torchaudio
import warnings
from pyannote.audio import Pipeline
from huggingface_hub import snapshot_download

# Tắt cảnh báo rác
warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")

MODEL_CACHE_DIR = r"C:\AI Project\Chatbot recording\model_detect"
os.makedirs(MODEL_CACHE_DIR, exist_ok=True)

def get_speaker_segments(audio_path, hf_token):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Đảm bảo model đã tải
    try:
        snapshot_download(
            repo_id="pyannote/speaker-diarization-3.1",
            token=hf_token,
            local_dir=MODEL_CACHE_DIR,
            local_dir_use_symlinks=False
        )
    except Exception as e:
        pass # Nếu đã tải rồi thì bỏ qua lỗi mạng

    # 2. Load Pipeline
    print(f"\n[Diarization] Đang khởi tạo Pipeline...")
    try:
        pipeline = Pipeline.from_pretrained(
            os.path.join(MODEL_CACHE_DIR, "config.yaml")
        ).to(device)
    except:
        pipeline = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            token=hf_token,
            cache_dir=MODEL_CACHE_DIR
        ).to(device)

    print(f"[Diarization] Đang phân tích file: {audio_path}...")

    # 3. Xử lý Audio
    waveform, sample_rate = torchaudio.load(audio_path)
    audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}
    
    # Run Inference
    output = pipeline(audio_in_memory)

    annotation = None

    # Trường hợp 1: Kết quả là Object chứa (Pyannote 3.1 mới)
    if hasattr(output, "speaker_diarization"):
        # Lấy Annotation thật sự từ bên trong object
        annotation = output.speaker_diarization
        
    # Trường hợp 2: Kết quả là Annotation trực tiếp (Pyannote cũ)
    elif hasattr(output, "itertracks"):
        annotation = output
        
    # Trường hợp 3: Kết quả là Tuple
    elif isinstance(output, tuple):
        annotation = output[0]

    segments = []
    
    if annotation:
        for turn, _, speaker in annotation.itertracks(yield_label=True):
            # Lọc các đoạn quá ngắn (< 0.5s)
            if turn.end - turn.start > 0.5:
                segments.append({
                    "start": turn.start,
                    "end": turn.end,
                    "speaker": speaker
                })
        print(f"[Diarization] Tìm thấy {len(segments)} lượt nói.")
    else:
        print("[Lỗi] Không tìm thấy dữ liệu Annotation trong kết quả trả về!")
        print(f"Structure: {dir(output)}")

    # Dọn dẹp
    del pipeline
    torch.cuda.empty_cache()
    gc.collect()
    
    return segments

if __name__ == "__main__":
    print("Module my_diarization đã sẵn sàng.")