import nemo.collections.asr as nemo_asr
import torch

MODEL_PATH = r"C:\AI Project\Chatbot recording\models\parakeet-ctc-0.6b-vi.nemo"

def load_asr_model():
    print(f"\n[ASR] Đang load model Parakeet từ: {MODEL_PATH}")
    
    # Load model
    asr_model = nemo_asr.models.EncDecCTCModel.restore_from(MODEL_PATH)
    
    # Chuyển sang GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    asr_model = asr_model.to(device)
    
    print(f"[ASR] Model đã sẵn sàng trên {device}!")
    return asr_model, device