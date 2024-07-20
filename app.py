import os
from modules import (
    AppPaths,
    ModelLoader,
)
from routes import register_routes,resources
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from models.requests.base_request import BaseRequest
import uvicorn

app=FastAPI();
origins = [
    "http://localhost:5000",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

APP_DIR=AppPaths.APP_DIR;
STATIC_FOLDER=AppPaths.STATIC_FOLDER;

# create processing folders for each model
for model in ModelLoader.detectors:
    os.makedirs(os.path.join(STATIC_FOLDER, model), exist_ok=True)

register_routes(app)
#DEBUG ONLY REMOVE IN PRODUCTION
@app.get("/api/delete")
def delete_embeddings():
    # delete all the saved databases states
    manager=resources.manager
    manager.delete_all()
    return {"result": "success"}




@app.get("/api/gallery")
def get_gallery(request:BaseRequest):
    manager=resources.manager
    detector_name = request.detector_name
    embedder_name = request.embedder_name

    embeddings=manager.get_all_embeddings(detector_name,embedder_name,False)
    result= [e.name for e in embeddings]
    return result




resources.init_resources();

for model_name, _ in ModelLoader.embedders.items():
    ModelLoader.load_embedder(model_name, APP_DIR)

ModelLoader.load_genderage("MobileNetCeleb0.25_CelebA", APP_DIR)

if __name__ == "__main__":
    try:
        uvicorn.run(app, host="0.0.0.0", port=5057)
    except Exception as e:
        print(f"Error: {e}")
