import os
from modules import (
    AppPaths,
    ModelLoader,
)
from fastapi.staticfiles import StaticFiles
from routes import register_routes,resources
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI
from models import EmbedderName,DetectorName
import uvicorn
from middlewares import register_middlewares

app=FastAPI();
origins = [
    "http://localhost:5000",
    "http://127.0.0.1:5000",
    "https://127.0.0.1:7101",
    "https://localhost:7101",
    "http://localhost:5156",
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
register_middlewares(app)
register_routes(app)
#DEBUG ONLY REMOVE IN PRODUCTION
@app.get("/api/delete")
def delete_embeddings():
    # delete all the saved databases states
    manager=resources.manager
    manager.delete_all()
    return {"result": "success"}






resources.init_resources();

app.mount("/static",StaticFiles(directory="static"),name="static");
app.mount("/pool",StaticFiles(directory="pool"),name="pool");


if __name__ == "__main__":
    try:
        uvicorn.run("app:app", host="0.0.0.0", port=5057,workers=1)
    except Exception as e:
        print(f"Error: {e}")
