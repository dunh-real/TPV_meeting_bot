from transformers import AutoProcessor, AutoModelForImageTextToText
import torch
import time

device = torch.device("cuda")
MODEL_PATH = "zai-org/GLM-OCR"
messages = [
    {
        "role": "user",
        "content": [
            {
                "type": "image",
                "url": "./1.png"
            },
            {
                "type": "text",
                "text": "Text Recognition:"
            }
        ],
    }
]
processor = AutoProcessor.from_pretrained(MODEL_PATH)
# model = AutoModelForImageTextToText.from_pretrained(
#     pretrained_model_name_or_path = MODEL_PATH,
#     torch_dtype = "auto",
#     device_map = "auto",
# )
model = AutoModelForImageTextToText.from_pretrained(MODEL_PATH)
model.to(device)

start_time = time.perf_counter()

inputs = processor.apply_chat_template(
    messages,
    tokenize = True,
    add_generation_prompt = True,
    return_dict = True,
    return_tensors = "pt"
).to(model.device)
inputs.pop("token_type_ids", None)
generated_ids = model.generate(**inputs, max_new_tokens = 8192)
output_text = processor.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens = False)

end_time = time.perf_counter()
ex_time = end_time - start_time
print(f"{ex_time:.4f} seconds")
print('-'*70)
print(output_text)