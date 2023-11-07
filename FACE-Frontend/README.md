# Facial Analysis and Comparison Engine (FACE)

## Description

This project involves the usage of flask to serve the UI and it communicates with an api that can recognize faces and calculate similarity.  
you can can the git respository for the api [here](https://github.com/ofekhta1/image_rec)

### Features

- [x] Upload images to the server
- [x] Compare two images and get evaluated similarity
- [x] Find most similar image from folder
- [x] Specify face for comparison in multi-face images

## Installation

### Prerequisites

Before you begin, ensure you have the following installed:

- Python 3.x
- [Pip](https://pip.pypa.io/en/stable/installation/) (Python package installer)
- [Virtualenv](https://virtualenv.pypa.io/en/latest/installation.html) (optional but recommended)

### Clone the Repository

```bash
git clone https://github.com/DanielGrin31/FACE-Frontend.git
cd FACE-Frontend
```

### Set Up Virtual Environment (Optional but Recommended)

Setting up a virtual environment is recommended to keep your dependencies isolated.

```bash
# Create a virtual environment
python3 -m venv venv

# Activate the virtual environment
# On Windows, use: venv\Scripts\activate
# On macOS and Linux, use: source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run the Application

```bash
# Set the FLASK_APP environment variable
# On Windows, use: set FLASK_APP=app.py
# On macOS and Linux, use: export FLASK_APP=app.py

# Enable debug mode (optional, recommended for development)
# On Windows, use: set FLASK_ENV=development
# On macOS and Linux, use: export FLASK_ENV=development

flask run
```

Visit `http://localhost:5000` in your web browser to access the application.
