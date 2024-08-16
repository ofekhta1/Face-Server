from services.models import ModelLoader,TritonModelLoader,LocalModelLoader
from services.stores import ImageGroupRepository,InMemoryImageEmbeddingManager,MetadataManager,MilvusImageEmbeddingManager
from config.app_paths import AppPaths
from config.config import StoreType,ProcessingType
from services.processing.face_aligner import FaceAligner
from services.processing.local.local_embedding_generator import LocalEmbeddingGenerator
from services.processing.face_similarity_search import FaceSimilaritySearch
from services.processing.face_clustering import FaceClustering
from services.processing.local.local_image_processor import LocalImageProcessor
from services.processing.triton.triton_image_processor import TritonImageProcessor
from services.processing.local.local_face_extractor import LocalFaceExtractor
from services.processing.triton.triton_face_extractor import TritonFaceExtractor
from services.processing.triton.triton_embedding_generator import TritonEmbeddingGenerator
from config.config import Settings
from models.stored_embedding import StoredEmbeddings
from dependency_injector import containers,providers
from services.processing.triton.triton_client_handler import TritonClientHandler
import os

class Container(containers.DeclarativeContainer):
    config= providers.Configuration();
    model_loaders=providers.Aggregate({
        ProcessingType.Local: providers.Singleton(LocalModelLoader),
        ProcessingType.Triton: providers.Singleton(TritonModelLoader),
    }) 
    default_model_loader=providers.Selector(
        config.Processing.Type,
        local= providers.Singleton(LocalModelLoader),
        mixed= providers.Singleton(LocalModelLoader),
        triton= providers.Singleton(TritonModelLoader),
    )


    emb_manager = providers.Selector(
        config.Store.Type,
        memory=providers.Singleton(InMemoryImageEmbeddingManager,AppPaths.APP_DIR,default_model_loader),
        milvus=providers.Singleton(MilvusImageEmbeddingManager, url=config.Store.URL,model_loader=default_model_loader)
    )

    groups=providers.Singleton(ImageGroupRepository,AppPaths.APP_DIR,default_model_loader)
    face_extractor=providers.Selector(
        config.Processing.Type,
        local=providers.Singleton(LocalFaceExtractor),
        mixed= providers.Singleton(LocalFaceExtractor),
        triton= providers.Singleton(TritonFaceExtractor),
    )
    embedding_generator=providers.Selector(
        config.Processing.Type,
        local=providers.Singleton(LocalEmbeddingGenerator,emb_manager,face_extractor),
        mixed=providers.Singleton(LocalEmbeddingGenerator,emb_manager,face_extractor),
        triton=providers.Singleton(TritonEmbeddingGenerator,emb_manager,face_extractor),
    )

    face_aligner = providers.Singleton(FaceAligner,face_extractor)
    face_clustering=providers.Singleton(FaceClustering,emb_manager,groups)
    face_similarity_search=providers.Singleton(FaceSimilaritySearch,emb_manager,embedding_generator,face_aligner);
    metadata_manager=providers.Singleton(MetadataManager,emb_manager,face_aligner,embedding_generator,default_model_loader);


    default_image_processor=providers.Selector(
        config.Processing.Type,
        local= providers.Singleton(LocalImageProcessor,emb_manager,face_aligner,embedding_generator,default_model_loader),
        mixed= providers.Singleton(LocalImageProcessor,emb_manager,face_aligner,embedding_generator,default_model_loader),
        triton= providers.Singleton(TritonImageProcessor,emb_manager,face_aligner,face_extractor,default_model_loader),
    )
async def init_resources(app):
    # global groups,helper,manager,cfg
    container=Container()
    app.container=container
    settings=Settings();
    
    container.config.from_dict(settings.model_dump())
    container.init_resources()
    TritonClientHandler.init(container.config.Processing.TritonURL())
    model_loader=container.default_model_loader()
    modules=[f"routes.{m.split('.')[0]}" for m in os.listdir("routes") if not m.startswith("__")]
    modules.append("app")
    container.wire(modules=modules)

    for model_name, _ in model_loader.detectors.items():
        model_loader.load_detector(model_name, AppPaths.APP_DIR)

    for model_name, _ in model_loader.embedders.items():
        model_loader.load_embedder(model_name, AppPaths.APP_DIR)

    model_loader.load_genderage("MobileNetCeleb0.25_CelebA", AppPaths.APP_DIR)
    
    manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager=container.emb_manager();

    for model_name, _ in model_loader.detectors.items():
        model_loader.load_detector(model_name, AppPaths.APP_DIR)
        manager.load(model_name)
        if settings.Store.Type==StoreType.Memory:
            for embedder_name, _ in model_loader.embedders.items():
                if embedder_name not in manager.db_embeddings[model_name].embeddings:
                    manager.db_embeddings[model_name].embeddings[embedder_name] = (
                        StoredEmbeddings([])
                    )