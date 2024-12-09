
import sys
sys.path.append("../")
from contextlib import asynccontextmanager
from config.app_paths import AppPaths
from routes import register_routes,resources
from fastapi.middleware.cors import CORSMiddleware
from fastapi import FastAPI,Depends
from dependency_injector.wiring import inject, Provide
from routes.resources import Container
import uvicorn
from middlewares import register_middlewares


@asynccontextmanager
async def lifespan(app: FastAPI):
    # OnStartup
    await resources.init_resources(app=app);

    yield
    # OnShutdown
    pass;





app=FastAPI(lifespan=lifespan);
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

register_middlewares(app)
register_routes(app)

#TODO:DEBUG ONLY REMOVE IN PRODUCTION
@app.get("/api/delete")
@inject

def delete_embeddings(manager=Depends(Provide[Container.emb_manager])):
    # delete all the saved databases states
    manager.delete_all()
    return {"result": "success"}








if __name__ == "__main__":
    try:

        uvicorn.run("app:app", host="0.0.0.0", port=5057)
    except Exception as e:
        print(f"Error: {e}")
