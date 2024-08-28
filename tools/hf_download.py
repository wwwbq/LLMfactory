from huggingface_hub import snapshot_download
import os


repo_type = "model" # dataset
repo_id = "llava-hf/llava-1.5-7b-hf"
local_dir = "/root/HF_HOME/llava-1.5-7b-hf"

if not os.path.exists(local_dir):
    os.makedirs(local_dir, exist_ok=True)


snapshot_download(repo_id=repo_id, local_dir=local_dir, repo_type=repo_type)