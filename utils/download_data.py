from huggingface_hub import snapshot_download

local_path = snapshot_download(
    repo_id="MartinJYHuang/multiturn-reasoning-attack", 
    repo_type="dataset",
    local_dir="./data",
    local_dir_use_symlinks=False  
)