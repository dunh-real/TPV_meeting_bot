import torchaudio

from my_diarization import get_speaker_segments
from load_paraket import load_asr_model

# --- CẤU HÌNH ---
HF_TOKEN = "hf_"
AUDIO_PATH = r"C:\AI Project\Chatbot recording\data_output\test.wav"
OUTPUT_FILE = r"C:\AI Project\Chatbot recording\Text\transcription2.txt"

def main():
    # BƯỚC 1: Lấy thông tin người nói (Pyannote xong tắt luôn)
    segments = get_speaker_segments(AUDIO_PATH, HF_TOKEN)

    # BƯỚC 2: Load model ASR
    asr_model, device = load_asr_model()

    # BƯỚC 3: Xử lý audio gốc
    print("[Main] Đang xử lý audio để cắt...")
    waveform, sample_rate = torchaudio.load(AUDIO_PATH)
    
    # Resample về 16k nếu cần
    if sample_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
        waveform = resampler(waveform)
    
    # Ép về Mono [1, time]
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    
    # Chuyển toàn bộ audio lên GPU một lần (hoặc để CPU cắt cho an toàn bộ nhớ nếu file quá dài)
    waveform = waveform.to(device)
    
    full_results = []
    
    print("\n[Main] Bắt đầu transcribe từng đoạn...")
    for i, seg in enumerate(segments):
        # Tính toán vị trí cắt (sample = giây * 16000)
        start_sample = int(seg['start'] * 16000)
        end_sample = int(seg['end'] * 16000)
        
        # Cắt audio: waveform shape là [1, time]
        # Lấy từ sample bắt đầu đến kết thúc
        chunk = waveform[0, start_sample:end_sample]
        
        # Đưa về CPU trước khi ném vào hàm transcribe
        chunk_cpu = chunk.cpu()
        
        try:
            # Transcribe
            trans = asr_model.transcribe(
                audio=[chunk_cpu],
                batch_size=1,
                num_workers=0,
                verbose=False
            )
            
            # Lấy text
            if hasattr(trans[0], 'text'):
                text = trans[0].text
            else:
                text = str(trans[0])
            
            text = text.strip()
            
            # Format kết quả
            line = f"[{seg['start']:.2f}s - {seg['end']:.2f}s] {seg['speaker']}: {text}"
            print(line)
            full_results.append(line)
            
        except Exception as e:
            print(f"Lỗi đoạn {i}: {e}")

    # BƯỚC 4: Lưu file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write("\n".join(full_results))
    print(f"\n[Done] Đã lưu kết quả vào {OUTPUT_FILE}")

if __name__ == "__main__":
    main()