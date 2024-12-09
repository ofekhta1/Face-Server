from fastapi import FastAPI
# Import your blueprints
from .file_handling import file_handling_router
from .clustering import clustering_router
from .image_metadata import image_metadata_router
from .image_processing import image_processing_router
from .image_search import image_search_router
from .metrics import metrics_router
from .batch_jobs import batch_jobs_router
def register_routes(app: FastAPI):
    # Register routes
    app.include_router(file_handling_router);
    app.include_router(image_search_router);
    app.include_router(image_metadata_router);
    app.include_router(clustering_router);
    app.include_router(image_processing_router);
    app.include_router(metrics_router);
    app.include_router(batch_jobs_router);