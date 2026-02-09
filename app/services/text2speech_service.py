import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from neucodec import NeuCodec
from phonemizer.backend import EspeakBackend
from vinorm import TTSnorm
import soundfile as sf
import numpy as np

# Load model
model_id = "dinhthuan/neutts-air-vi"  
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
).to("cuda")
model.eval()

# Load codec
codec = NeuCodec.from_pretrained("neuphonic/neucodec").to("cuda")
codec.eval()

# Initialize phonemizer
phonemizer = EspeakBackend(language='vi', preserve_punctuation=True, with_stress=True)

# Normalize and phonemize text
text = "Xin chào, đây là mô hình text to speech tiếng Việt"
text_normalized = TTSnorm(text, punc=False, unknown=True, lower=False, rule=False)
phones = phonemizer.phonemize([text_normalized])[0]

# Encode reference audio (for voice cloning)
from librosa import load as librosa_load
ref_audio_path = "reference.wav"
ref_text = "Đây là văn bản tham chiếu"
ref_text_normalized = TTSnorm(ref_text, punc=False, unknown=True, lower=False, rule=False)
ref_phones = phonemizer.phonemize([ref_text_normalized])[0]

wav, _ = librosa_load(ref_audio_path, sr=16000, mono=True)
wav_tensor = torch.from_numpy(wav).float().unsqueeze(0).unsqueeze(0)
with torch.no_grad():
    ref_codes = codec.encode_code(audio_or_path=wav_tensor).squeeze(0).squeeze(0).cpu()

# Generate speech
codes_str = "".join([f"<|speech_{i}|>" for i in ref_codes.tolist()])
combined_phones = ref_phones + " " + phones
chat = f"""user: Convert the text to speech:<|TEXT_PROMPT_START|>{combined_phones}<|TEXT_PROMPT_END|>\nassistant:<|SPEECH_GENERATION_START|>{codes_str}"""

input_ids = tokenizer.encode(chat, return_tensors="pt").to("cuda")
speech_end_id = tokenizer.convert_tokens_to_ids("<|SPEECH_GENERATION_END|>")

with torch.no_grad():
    output = model.generate(
        input_ids,
        max_new_tokens=2048,
        temperature=1.0,
        top_k=50,
        eos_token_id=speech_end_id,
        pad_token_id=tokenizer.eos_token_id,
    )

# Decode to audio
output_text = tokenizer.decode(output[0], skip_special_tokens=False)
# Extract speech codes and decode with codec...
# (See full implementation in repository)

# Save audio
sf.write("output.wav", audio, 24000)