import git
import subprocess
import os
import sys
import gdown
import zipfile

major,minor=0
def clone_repo(repo_url, destination):
    git.Repo.clone_from(repo_url, destination)


def install_requirements(repo_path):

    if os.name == "nt":  # For Windows
        pip_path = ".venv\\Scripts\\pip"
    else:  # For Unix-like systems
        pip_path = ".venv/bin/pip3"
    subprocess.run(
        [pip_path, "install", "-r", os.path.join(repo_path, "requirements.txt")]
    )


def check_python_version():
    global major,minor
    major, minor = sys.version_info[:2]
    if major != 3 or minor<10 or minor >11:
        print(
            "Error: This script requires Python version 3.x, 3.10-3.11 is recommended"
        )
        sys.exit(1)
def download_models_gdrive(model_name:str,model_file_id:str,save_path:str):
        os.makedirs(save_path,exist_ok=True);
        download_path = os.path.join("Face_Server","OnnxModels",model_name+".zip")
        if not os.path.exists(download_path):
            print(f"downloading {model_name} into {download_path}")
            gdown.download(id=model_file_id, output=download_path, quiet=False)
        print(f"extracting {model_name}")
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(save_path)
def download_models():
    model_ids={
        "Embedders":"1lnHB3FFLXOenJ-KCnMvKibcj0y7CY2e4",
        "Detectors":"1gm8KET04Q-ZGIfAX9M0dHUtHPuSRK8EW",
        "GenderAge":"1kEil84PS1Sp_yAQ9SLAVbeNzdLl5Peu3"
    }
    for model in model_ids:
        download_models_gdrive(model,model_ids[model],os.path.join("Face_Server", "OnnxModels", model))


def main():
    # Clone the frontned repository
    if not os.path.exists("Face_Frontend"):
        print("Cloning Face-Frontend...")
        clone_repo(
            "https://github.com/ofekhta1/Face-Frontend",
            "Face_Frontend",
        )
        print("Face-Frontend cloned successfully.")

    if not os.path.exists("Face_Server"):
        # Clone the second repository
        print("Cloning Face-Server...")
        clone_repo(
            "https://github.com/ofekhta1/Face-Server", "Face_Server"
        )
        print("Face-Server cloned successfully.")

    # Create a virtual environment
    if not os.path.exists(".venv"):
        print("Creating virtual environment...")
        subprocess.run([sys.executable, "-m", "venv", ".venv"])
        print("Virtual environment created successfully.")

    # Install requirements for the first repository
    print("Installing requirements for Face_Frontend...")
    install_requirements("Face_Frontend")
    print("Requirements for Face_Frontend installed successfully.")

    # Install requirements for the second repository
    print("Installing requirements for Face_Server...")
    install_requirements("Face_Server")
    print("Requirements for Face_Server installed successfully.")

    download_models()


if __name__ == "__main__":
    main()
