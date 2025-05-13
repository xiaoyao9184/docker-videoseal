import os
import re
import sys
import git
import subprocess
from huggingface_hub import hf_hub_download

REPO_URL = "https://github.com/facebookresearch/videoseal.git"
REPO_BRANCH = '3493bd3c39679e6264b4c09dd427ae024246d6b5'
LOCAL_PATH = "./videoseal"

def install_src():
    if not os.path.exists(LOCAL_PATH):
        print(f"Cloning repository from {REPO_URL}...")
        repo = git.Repo.clone_from(REPO_URL, LOCAL_PATH)
        repo.git.checkout(REPO_BRANCH)
    else:
        print(f"Repository already exists at {LOCAL_PATH}")

    requirements_path = os.path.join(LOCAL_PATH, "requirements.txt")
    if os.path.exists(requirements_path):
        print("Installing requirements...")
        # Lock versions by attaching a custom requirements.txt.
        subprocess.check_call(["pip", "install", "-r", "./requirements.txt", "-r", requirements_path])
    else:
        print("No requirements.txt found.")

# clone repo
install_src()

# set app path form import
os.environ["APP_PATH"] = LOCAL_PATH

# run gradio
# from gradio.cli import cli
# if __name__ == '__main__':
#     sys.argv[0] = re.sub(r'(-script\.pyw|\.exe)?$', '', sys.argv[0]) ##os.path.join(os.path.dirname(sys.argv[0]), "gradio_run.py")
#     sys.argv.append(os.path.join(os.path.dirname(sys.argv[0]), "gradio_app.py"))
#     sys.exit(cli())

subprocess.check_call(["python", os.path.join(os.path.dirname(sys.argv[0]), "gradio_app.py")])
