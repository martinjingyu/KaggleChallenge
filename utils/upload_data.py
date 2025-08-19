from huggingface_hub import HfApi
import os

def upload_data():
    api = HfApi(token="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")
    api.upload_folder(
        folder_path="data",
        path_in_repo=".",
        repo_id="MartinJYHuang/multiturn-reasoning-attack",
        repo_type="dataset",
        token="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"
    )
def upload_single_data(folder_path, file_name):
    api = HfApi(token="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")
    
    api.upload_file(
        path_or_fileobj=os.path.join(folder_path, file_name),
        path_in_repo=os.path.relpath(folder_path, "data"),
        repo_id="MartinJYHuang/multiturn-reasoning-attack",
        repo_type="dataset",
        token="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"
    )
# api.upload_large_folder(
#     folder_path="mcts/mcts_data",  
#     repo_id="MartinJYHuang/MultiturnJailbreak",     
#     repo_type="dataset",                    
#     path_in_repo="./",                      
#     commit_message="Upload large folder"
# )
if __name__ == "__main__":
    upload_single_data("./data/raw_data/test/rollout5","tree.json")