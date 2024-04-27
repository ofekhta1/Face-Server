import git
import subprocess
import os
import sys
import gdown
import zipfile

def clone_repo(repo_url, destination):
    git.Repo.clone_from(repo_url, destination)

def install_requirements(repo_path):
    subprocess.run([".venv/bin/pip3" , 'install', '-r', os.path.join(repo_path, 'requirements.txt')])

def check_python_version():
    major, minor = sys.version_info[:2]
    if major!=3:
        print("Error: This script requires Python version 3.x, 3.10-3.12 is recommended")
        sys.exit(1)
def download_models():
    #download antelopev2
    model_file_ids={
        "antelopev2":"18wEUfMNohBJ4K3Ly5wpTejPfDzp-8fI8",
        "buffalo_l":"1qXsQJ8ZT42_xSmWIYy85IcidpiZudOCB"
    }
    for model in model_file_ids:
        download_path=os.path.join("Face_Server","OnnxModels",model+".zip");
        if(not os.path.exists(download_path)):
            print(f"downloading {model} into {download_path}")
            gdown.download(id=model_file_ids[model],
                    output=download_path, quiet=False)
        if(len(os.listdir(os.path.join("Face_Server","OnnxModels",model)))<=1):
            print(f"extracting {model}")
            with zipfile.ZipFile(download_path,'r') as zip_ref:
                zip_ref.extractall(os.path.join("Face_Server","OnnxModels"))

        


def main():
    # Clone the frontned repository
    if(not os.path.exists("Face_Frontend")):
        print("Cloning Face-Frontend...")
        clone_repo("https://github.com/ofekhta1/FACE-Frontend_new_ofek_version", "Face_Frontend")
        print("Face-Frontend cloned successfully.")

    if(not os.path.exists("Face_Server")):
        # Clone the second repository
        print("Cloning Face-Server...")
        clone_repo("https://github.com/ofekhta1/image_rec_new_ofek_version", "Face_Server")
        print("Face-Server cloned successfully.")

    # Create a virtual environment
    if(not os.path.exists(".venv")):
        print("Creating virtual environment...")
        subprocess.run(['python3', '-m', 'venv', '.venv'])
        print("Virtual environment created successfully.")

    # Activate the virtual environment
    if os.name == 'nt':  # For Windows
        activate_script = '.venv\\Scripts\\activate'
    else:  # For Unix-like systems
        activate_script = '.venv/bin/activate'

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