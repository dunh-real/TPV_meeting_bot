from huggingface_hub import snapshot_download
import os
from dotenv import load_dotenv

load_dotenv()

MODEL_DETECTION_PATH = os.getenv("MODEL_DETECTION_PATH")
HF_TOKEN = os.getenv("HF_TOKEN")

def ensure_diarization_model(model_id: str = "pyannote/speaker-diarization-3.1"):
    os.makedirs(MODEL_DETECTION_PATH, exist_ok=True)
    if not os.path.exists(os.path.join(MODEL_DETECTION_PATH, "config.yaml")):
        print(f"[Diarization] Đang tải model {model_id}...")
        try:
            snapshot_download(
                repo_id=model_id,
                token=HF_TOKEN,
                local_dir=MODEL_DETECTION_PATH,
                local_dir_use_symlinks=False
            )
            print(f"[Diarization] Model đã sẵn sàng tại {MODEL_DETECTION_PATH}")
        except Exception as e:
            raise RuntimeError(f"[Diarization] Tải model thất bại: {e}")
    else:
        print(f"[Diarization] Model đã có sẵn tại {MODEL_DETECTION_PATH}")