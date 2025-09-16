from huggingface_hub import snapshot_download
path = snapshot_download(
    repo_id="pixparse/idl-wds",
    repo_type="dataset",
    allow_patterns="idl-train-00000.tar",
    local_dir="../data/idl_data",
    local_dir_use_symlinks=False
)
print(f"Downloaded to: {path}")