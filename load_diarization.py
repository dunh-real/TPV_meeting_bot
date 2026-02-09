import torch
import gc
import os
import torchaudio
import warnings
from pyannote.audio import Pipeline
from huggingface_hub import snapshot_download
from dotenv import load_dotenv

load_dotenv()

warnings.filterwarnings("ignore", message=".*torchcodec.*")
warnings.filterwarnings("ignore", message=".*TensorFloat-32.*")

MODEL_DETECTION_PATH = os.getenv("MODEL_DETECTION_PATH")
os.makedirs(MODEL_DETECTION_PATH, exist_ok=True)

class DiarizationProcessor:

    def __init__(self) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def load_pipeline(self):

        # Load pipeline diarization
        print("[Diarization] Đang khởi tạo pipeline diarization...")

        pipeline = Pipeline.from_pretrained(
            os.path.join(MODEL_DETECTION_PATH, "config.yaml"),
        ).to(self.device)

        print("[Diarization] Pipeline diarization đã sẵn sàng.")
        return pipeline
    
    # process audio and get speaker segments
    def process_audio(self, wav_input: str, pipeline):
        print(f"[Diarization] Đang phân tích file: {wav_input}...")

        waveform, sample_rate = torchaudio.load(wav_input)
        audio_in_memory = {"waveform": waveform, "sample_rate": sample_rate}

        output = pipeline(audio_in_memory)

        annotation = None
        if hasattr(output, "speaker_diarization"):
            annotation = output.speaker_diarization  # nếu là DiarizeOutput, lấy Annotation bên trong
        elif hasattr(output, "itertracks"):
            annotation = output
        elif isinstance(output, tuple):
            annotation = output[0]

        segments = []
        if annotation:
            for turn, _, speaker in annotation.itertracks(yield_label=True):  # dùng annotation
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

        # Giải phóng bộ nhớ
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

        return segments
    
diarization_processor = DiarizationProcessor()