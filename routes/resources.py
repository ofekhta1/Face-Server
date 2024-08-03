from services import ModelLoader
from services.stores import ImageGroupRepository,InMemoryImageEmbeddingManager,MetadataManager,MilvusImageEmbeddingManager
from config.app_paths import AppPaths
from services.processing.local import *
from services.processing.triton import *
from services.processing import FaceAligner,FaceClustering,FaceSimilaritySearch
from config.config import Settings
from models.stored_embedding import StoredEmbeddings
from dependency_injector import containers,providers
import os

class Container(containers.DeclarativeContainer):
    config= providers.Configuration();

    emb_manager = providers.Selector(
        config.Store.Type,
        memory=providers.Singleton(InMemoryImageEmbeddingManager,AppPaths.APP_DIR),
        milvus=providers.Singleton(MilvusImageEmbeddingManager, url=config.Store.URL)
    )

    groups=providers.Singleton(ImageGroupRepository,AppPaths.APP_DIR)

    face_extractor = providers.Selector(
        config.Processing,
        local=providers.Singleton(LocalFaceExtractor),
        triton=providers.Singleton(TritonFaceExtractor)
    )
    embedding_generator = providers.Selector(
        config.Processing,
        local=providers.Singleton(LocalEmbeddingGenerator,emb_manager,face_extractor),
        triton=providers.Singleton(TritonEmbeddingGenerator,TritonFaceExtractor)
    )
    
    face_aligner = providers.Singleton(FaceAligner,face_extractor)
    face_clustering=providers.Singleton(FaceClustering,emb_manager,groups)
    face_similarity_search=providers.Singleton(FaceSimilaritySearch,emb_manager,embedding_generator,face_aligner);
    metadata_manager=providers.Singleton(MetadataManager,emb_manager,face_aligner,embedding_generator);


async def init_resources(app):
    # global groups,helper,manager,cfg
    container=Container()
    app.container=container
    settings=Settings();
    container.config.from_dict(settings.model_dump())
    container.init_resources()
    modules=[f"routes.{m.split('.')[0]}" for m in os.listdir("routes") if not m.startswith("__")]
    modules.append("app")
    container.wire(modules=modules)

    for model_name, _ in ModelLoader.detectors.items():
        ModelLoader.load_detector(model_name, AppPaths.APP_DIR)

    for model_name, _ in ModelLoader.embedders.items():
        ModelLoader.load_embedder(model_name, AppPaths.APP_DIR)

    ModelLoader.load_genderage("MobileNetCeleb0.25_CelebA", AppPaths.APP_DIR)

    manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager=container.emb_manager();

    for model_name, _ in ModelLoader.detectors.items():
        ModelLoader.load_detector(model_name, AppPaths.APP_DIR)
        manager.load(model_name)
        for embedder_name, _ in ModelLoader.embedders.items():
            if embedder_name not in manager.db_embeddings[model_name].embeddings:
                manager.db_embeddings[model_name].embeddings[embedder_name] = (
                    StoredEmbeddings([])
                )