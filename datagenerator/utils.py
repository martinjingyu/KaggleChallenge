import sys
import os
from huggingface_hub import HfApi
sys.path.append(".")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../")))

token = ""
repo_id = ""

def upload_data():
    api = HfApi(token=token)
    api.upload_folder(
        folder_path="data",
        path_in_repo=".",
        repo_id=repo_id,
        repo_type="dataset",
    )
    
def upload_single_data(folder_path, file_name):
    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=os.path.join(folder_path, file_name),
        path_in_repo=os.path.join(os.path.relpath(folder_path, "data"),file_name),
        repo_id=repo_id,
        repo_type="dataset",
    )