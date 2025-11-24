from huggingface_hub import HfApi

api = HfApi()
repo_id = "oliveryql/gemma270m-sft-triviaqa" # change version number
api.create_repo(repo_id, repo_type="model", private=True, exist_ok=True)

api.upload_folder(
    folder_path="~/cs590llm/cs590-slm/models/gemma270m-sft-triviaqa",  # change model version here
    repo_id=repo_id,
    repo_type="model",
    path_in_repo=".",                # upload at repo root
    commit_message="Add SFT checkpoints",
)
print("Pushed to:", f"https://huggingface.co/{repo_id}")
