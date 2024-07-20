from flask import Flask
from fastapi import FastAPI
# Import your blueprints
from .file_handling import file_handling_router
from .clustering import image_clustering_bp
from .image_metadata import image_metadata_bp
from .image_processing import image_processing_bp
from .image_search import image_search_bp

def register_routes(app: FastAPI):
    # Register routes
    app.include_router(file_handling_router);
    
    # app.register_blueprint(file_handling_bp)
    # app.register_blueprint(image_clustering_bp)
    # app.register_blueprint(image_metadata_bp)
    # app.register_blueprint(image_processing_bp)
    # app.register_blueprint(image_search_bp)
