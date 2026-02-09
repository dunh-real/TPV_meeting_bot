import torchaudio
import torch
import os
import gc
import time 
from dotenv import load_dotenv

from convert_audio import converter
from load_parakeet import parakeet_loader
from load_diarization import diarization_processor
from down_diarization import ensure_diarization_model

load_dotenv()
OUTPUT_FILE = os.getenv("OUTPUT_FILE")
print(f"[Debug] OUTPUT_FILE đang dùng: {OUTPUT_FILE}")
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

class OutputGenerator:
    def __init__(self) -> None:
        ensure_diarization_model()

        self.asr_model, self.device = parakeet_loader.load_parakeet_model()
        self.diarization_pipeline = diarization_processor.load_pipeline()

    def convert(self, mp3_input: str):
        return converter.convert_audio(mp3_input)

    def segments(self, wav_input: str):
        return diarization_processor.process_audio(wav_input, self.diarization_pipeline)

    # squeeze channel dim và truyền audio dưới dạng list[[T]]
    def generate_output(self, mp3_input: str):
        total_start = time.time()

        print(f"[Timing] === Bắt đầu xử lý file: {mp3_input} ===")

        # 1. Chuyển đổi MP3 → WAV
        convert_start = time.time()
        wav_path = self.convert(mp3_input)
        convert_time = time.time() - convert_start
        print(f"[Timing] Chuyển đổi audio (MP3 → WAV): {convert_time:.2f}s")

        # 2. Diarization
        diar_start = time.time()
        segments = self.segments(wav_path)
        diar_time = time.time() - diar_start
        print(f"[Timing] Diarization (phân đoạn người nói): {diar_time:.2f}s | Tìm thấy {len(segments)} đoạn")

        # Chuẩn bị waveform
        waveform, sample_rate = torchaudio.load(wav_path)
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
        if waveform.shape[0] > 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)
        waveform = waveform.to(self.device)

        # 3. ASR 
        asr_start = time.time()
        full_results = []
        print("[Output] Đang xử lý phân đoạn và nhận dạng giọng nói...")
        for i, seg in enumerate(segments):
            start_sample = int(seg['start'] * 16000)
            end_sample = int(seg['end'] * 16000)

            # Tránh segment rỗng hoặc out of range
            if end_sample <= start_sample or end_sample > waveform.shape[1]:
                print(f"[Skip] Đoạn {i} quá ngắn hoặc lỗi giới hạn, bỏ qua.")
                text = ""
            else:
                # Lấy chunk và squeeze channel dim → shape [T]
                chunk = waveform[:, start_sample:end_sample].squeeze(0)  # [1, T] → [T]

                try:
                    # Truyền dưới dạng list để NeMo xử lý đúng raw waveform
                    trans = self.asr_model.transcribe(
                        audio=[chunk], # list[[T]]
                        batch_size=1,
                        num_workers=0
                    )

                    # Xử lý output
                    if isinstance(trans, list) and len(trans) > 0:
                        if hasattr(trans[0], 'text'):
                            text = trans[0].text
                        else:
                            text = str(trans[0])
                    else:
                        text = ""

                    text = text.strip()

                    line = f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}: {text}"
                    print(line)
                    full_results.append(line)

                except Exception as e:
                    print(f"Lỗi đoạn {i}: {e}")
                    full_results.append(f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}: [LỖI TRANSCRIBE]")

        asr_time = time.time() - asr_start
        print(f"[Timing] ASR (nhận dạng giọng nói): {asr_time:.2f}s")

        # Lưu kết quả
        with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
            f.write("\n".join(full_results))
        print(f"\n[Done] Đã lưu kết quả vào {OUTPUT_FILE}")

        # Tổng thời gian
        total_time = time.time() - total_start
        print(f"[Timing] === TỔNG THỜI GIAN XỬ LÝ FILE: {total_time:.2f}s ===\n")

        # Cleanup
        if os.path.exists(wav_path):
            os.remove(wav_path)
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        gc.collect()

if __name__ == "__main__":
    generator = OutputGenerator()
    while True:
        mp3_input = input("\nNhập đường dẫn file MP3 (hoặc 'exit' để thoát): ").strip()
        if mp3_input.lower() == "exit":
            print("[Exit] Đã thoát chương trình.")
            break
        if not os.path.exists(mp3_input):
            print(f"[Error] File không tồn tại: {mp3_input}")
            continue
        try:
            generator.generate_output(mp3_input)
        except Exception as e:
            print(f"[Error] Lỗi xử lý file: {e}")