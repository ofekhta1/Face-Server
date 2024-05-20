import git
import subprocess
import os
import sys
import gdown
import zipfile


def clone_repo(repo_url, destination):
    git.Repo.clone_from(repo_url, destination)


def install_requirements(repo_path):
    subprocess.run(
        [".venv/bin/pip3", "install", "-r", os.path.join(repo_path, "requirements.txt")]
    )


def check_python_version():
    major, minor = sys.version_info[:2]
    if major != 3:
        print(
            "Error: This script requires Python version 3.x, 3.10-3.12 is recommended"
        )
        sys.exit(1)
def download_model(model_name:str,model_file_id:str,save_path:str):
        os.makedirs(save_path,exist_ok=True);
        download_path = os.path.join("Face_Server", "OnnxModels", model_name + ".zip")
        if not os.path.exists(download_path):
            print(f"downloading {model_name} into {download_path}")
            gdown.download(id=model_file_id, output=download_path, quiet=False)
        print(f"extracting {model_name}")
        with zipfile.ZipFile(download_path, "r") as zip_ref:
            zip_ref.extractall(save_path)
def download_models():
    # download antelopev2
    embedder_file_ids = {
        "resnet50": "1N70r85EqHW9ghzgZPq5t2d0e-t__qpUJ",
        "resnet100": "1rxKC8b_lsvCnIr88W2lGNgMc2Afdw7wh",
    }
    detector_file_ids = {
        "retinanet": "1j4Dm8TguSTk7iCiwCxxeZNZf8Ey0UYF6",
        "scrfd": "1Gp9nI1M9jjYP9evRikkSV9RINQ4eVF30",
    }
    gender_age_file_ids = {
        "mobilenet_genderage": "",
    }
    for model in embedder_file_ids:
        download_model(model,embedder_file_ids[model],os.path.join("Face_Server", "OnnxModels", "Embedders"))

    for model in detector_file_ids:
        download_model(model,detector_file_ids[model],os.path.join("Face_Server", "OnnxModels", "Detectors"))

    for model in gender_age_file_ids:
        download_model(model,gender_age_file_ids[model],os.path.join("Face_Server", "OnnxModels", "GenderAge"))


def main():
    # Clone the frontned repository
    if not os.path.exists("Face_Frontend"):
        print("Cloning Face-Frontend...")
        clone_repo(
            "https://github.com/ofekhta1/FACE-Frontend_new_ofek_version",
            "Face_Frontend",
        )
        print("Face-Frontend cloned successfully.")

    if not os.path.exists("Face_Server"):
        # Clone the second repository
        print("Cloning Face-Server...")
        clone_repo(
            "https://github.com/ofekhta1/image_rec_new_ofek_version", "Face_Server"
        )
        print("Face-Server cloned successfully.")

    # Create a virtual environment
    if not os.path.exists(".venv"):
        print("Creating virtual environment...")
        subprocess.run(["python3", "-m", "venv", ".venv"])
        print("Virtual environment created successfully.")

    # Activate the virtual environment
    if os.name == "nt":  # For Windows
        activate_script = ".venv\\Scripts\\activate"
    else:  # For Unix-like systems
        activate_script = ".venv/bin/activate"

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
