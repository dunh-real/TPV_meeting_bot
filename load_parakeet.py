import nemo.collections.asr as nemo_asr
import torch
import os
from dotenv import load_dotenv

class ParakeetLoader:

    def __init__(self, model_path:  str) -> None:
        self.model_path = model_path

    def load_parakeet_model(self):

        # Check if GPU is available
        if torch.cuda.is_available():
            device = torch.device("cuda")
            print(f"Sử dụng {device} để load model ASR Paraket.")
        else:
            device = torch.device("cpu")
            print("GPU không khả dụng, sử dụng CPU để load model ASR Paraket.")

        print("[ASR] Đang load model Paraket...")

        # Load model Paraket
        asr_model = nemo_asr.models.EncDecCTCModel.restore_from(self.model_path, map_location=device)
        asr_model = asr_model.to(device)

        print(f"[ASR] Load model Paraket thành công, sẵn sàng để sử dụng trên {device}.")

        return asr_model, device
    
parakeet_loader = ParakeetLoader(model_path=os.getenv("PARAKET_MODEL_PATH"))