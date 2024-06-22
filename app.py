from flask import Flask, request, jsonify
import os
import cv2
import sys
from modules import (
    AppPaths,
    ModelLoader,
)
from routes import register_routes,resources
from flask_cors import CORS


# Define directories
APP_DIR = os.path.dirname(sys.argv[0])
UPLOAD_FOLDER = os.path.join(APP_DIR, "pool")
STATIC_FOLDER = os.path.join(APP_DIR, "static")

# create dirs
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, "no_face"), exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

# create processing folders for each model
for model in ModelLoader.detectors:
    os.makedirs(os.path.join(STATIC_FOLDER, model), exist_ok=True)

app = Flask(__name__, static_folder="static")
register_routes(app)
cors = CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/api/delete", methods=["GET"])
def delete_embeddings():
    # delete all the saved databases states
    manager=resources.manager
    manager.delete_all()
    return jsonify({"result": "success"})




@app.route("/api/gallery", methods=["GET"])
def get_gallery():
    manager=resources.manager
    detector_name = request.args.get("detector_name", default="SCRFD10G", type=str)
    embedder_name = request.args.get(
        "embedder_name", default="ResNet100GLint360K", type=str
    )
    embeddings=manager.get_all_embeddings(detector_name,embedder_name,False)
    result= [e.name for e in embeddings]
    return jsonify(result)



app.secret_key = "your_secret_key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ROOT_FOLDER"] = APP_DIR
AppPaths.APP_DIR=APP_DIR
AppPaths.STATIC_FOLDER=STATIC_FOLDER
AppPaths.UPLOAD_FOLDER=UPLOAD_FOLDER
resources.init_resources();

for model_name, _ in ModelLoader.embedders.items():
    ModelLoader.load_embedder(model_name, APP_DIR)

ModelLoader.load_genderage("MobileNetCeleb0.25_CelebA", APP_DIR)

if __name__ == "__main__":
    try:
        app.run(debug=True, port=5057, use_reloader=False)
    except Exception as e:
        print(f"Error: {e}")
