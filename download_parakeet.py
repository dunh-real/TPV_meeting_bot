from huggingface_hub import list_repo_files

files = list_repo_files("nvidia/parakeet-ctc-0.6b-Vietnamese")
for f in files:
    print(f)

from huggingface_hub import hf_hub_download

model_path = hf_hub_download(
    repo_id="nvidia/parakeet-ctc-0.6b-Vietnamese",
    filename="parakeet-ctc-0.6b-vi.nemo",
    local_dir="C:/AI Project/Chatbot recording/models",
    local_dir_use_symlinks=False
)

print("Downloaded to:", model_path)