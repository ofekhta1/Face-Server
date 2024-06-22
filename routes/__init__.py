from flask import Flask

# Import your blueprints
from .file_handling import file_handling_bp
from .clustering import image_clustering_bp
from .image_metadata import image_metadata_bp
from .image_processing import image_processing_bp
from .image_search import image_search_bp

def register_routes(app: Flask):
    # Register blueprints
    app.register_blueprint(file_handling_bp)
    app.register_blueprint(image_clustering_bp)
    app.register_blueprint(image_metadata_bp)
    app.register_blueprint(image_processing_bp)
    app.register_blueprint(image_search_bp)
