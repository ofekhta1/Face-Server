from services.models import ModelLoader,TritonModelLoader,LocalModelLoader
from services.stores import ImageGroupRepository,InMemoryImageEmbeddingManager,MetadataManager,MilvusImageEmbeddingManager
from config.app_paths import AppPaths
from config.config import EmbeddingStoreType,ImageStoreType,ProcessingType
from services.processing.face_aligner import FaceAligner
from services.processing.local.local_embedding_generator import LocalEmbeddingGenerator
from services.processing.face_similarity_search import FaceSimilaritySearch
from services.processing.face_clustering import FaceClustering
from services.processing.local.local_image_processor import LocalImageProcessor
from services.processing.triton.triton_image_processor import TritonImageProcessor
from services.processing.local.local_face_extractor import LocalFaceExtractor
from services.processing.triton.triton_face_extractor import TritonFaceExtractor
from services.stores.images import LocalImageStorage,MinioImageStorage
from services.processing.triton.triton_embedding_generator import TritonEmbeddingGenerator
from config.config import Settings
from models.stored_embedding import StoredEmbeddings
from services.logging import ConsoleLogger
from dependency_injector import containers,providers
from services.queue import InMemoryProcessingQueue,Consumer,RabbitMQQueue
from services.processing.triton.triton_client_handler import TritonClientHandler
import os
import sys
from fastapi.staticfiles import StaticFiles
from fastapi import FastAPI

class Container(containers.DeclarativeContainer):
    config= providers.Configuration();
    logger=providers.Singleton(ConsoleLogger,config.Logging.Verbose)
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
    image_storage=providers.Selector(config.Store.Image.Type,
                                     local=providers.Singleton(LocalImageStorage),
                                     minio=providers.Singleton(MinioImageStorage, endpoint=config.Store.Image.URL,
                                                                access_key=config.Store.Image.AccessKey,
                                                                secret_key=config.Store.Image.SecretKey,
                                                                logger=logger))

    emb_manager = providers.Selector(
        config.Store.Embedding.Type,
        memory=providers.Singleton(InMemoryImageEmbeddingManager,AppPaths.APP_DIR,default_model_loader),
        milvus=providers.Singleton(MilvusImageEmbeddingManager, url=config.Store.Embedding.URL,model_loader=default_model_loader)
    )

    groups=providers.Singleton(ImageGroupRepository,AppPaths.APP_DIR,default_model_loader,logger)
    face_extractor=providers.Selector(
        config.Processing.Type,
        local=providers.Singleton(LocalFaceExtractor,default_model_loader),
        mixed= providers.Singleton(LocalFaceExtractor,default_model_loader),
        triton= providers.Singleton(TritonFaceExtractor,default_model_loader),
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
    queue=providers.Selector(
        config.Queue.Type,
        memory = providers.Singleton(InMemoryProcessingQueue),
        rabbitmq = providers.Singleton(RabbitMQQueue,config.Queue.Host,config.Queue.Port)
    )
    default_image_processor=providers.Selector(
        config.Processing.Type,
        local= providers.Singleton(LocalImageProcessor,emb_manager,face_aligner,embedding_generator,default_model_loader),
        mixed= providers.Singleton(LocalImageProcessor,emb_manager,face_aligner,embedding_generator,default_model_loader),
        triton= providers.Singleton(TritonImageProcessor,emb_manager,face_aligner,face_extractor,default_model_loader),
    )
    consumer=providers.Selector(
        config.Queue.Type,
        memory = providers.Singleton(Consumer, queue,emb_manager,default_image_processor,image_storage),
        rabbitmq = providers.Singleton(Consumer, queue,emb_manager,default_image_processor,image_storage,config.Queue.URL)
    )


async def init_resources(app):
    # global groups,helper,manager,cfg
    container=Container()
    app.container=container
    settings=Settings();
    
    container.config.from_dict(settings.model_dump())
    container.init_resources()

    TritonClientHandler.init(container.config.Processing.TritonURL())
    model_loader:ModelLoader=container.default_model_loader()
    modules=[f"routes.{m.split('.')[0]}" for m in os.listdir("routes") if not m.startswith("__")]
    modules.append("app")
    container.wire(modules=modules)

    init_storage(app,container,settings)

    for model_name, _ in model_loader.model_registry["detectors"].items():
        model_loader.load_detector(model_name, AppPaths.APP_DIR)

    for model_name, _ in model_loader.model_registry["embedders"].items():
        model_loader.load_embedder(model_name, AppPaths.APP_DIR)

    model_loader.load_genderage("MobileNetCeleb0.25_CelebA", AppPaths.APP_DIR)
    
    manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager=container.emb_manager();
    if settings.Queue.Consume:
        consumer=container.consumer();
    
    for model_name, _ in model_loader.model_registry["detectors"].items():
        model_loader.load_detector(model_name, AppPaths.APP_DIR)
        manager.load(model_name)
        if settings.Store.Embedding.Type ==EmbeddingStoreType.Memory:
            for embedder_name, _ in model_loader.model_registry["embedders"].items():
                if embedder_name not in manager.db_embeddings[model_name].embeddings:
                    manager.db_embeddings[model_name].embeddings[embedder_name] = (
                        StoredEmbeddings([])
                    )


def init_storage(app:FastAPI,container:Container,settings:Settings):
    image_storage:LocalImageStorage|MinioImageStorage=container.image_storage()
    APP_DIR = os.path.dirname(sys.argv[0])
    UPLOAD_FOLDER = os.path.join(APP_DIR, "pool")
    STATIC_FOLDER = os.path.join(APP_DIR, "static")

    # create dirs
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    os.makedirs(os.path.join(UPLOAD_FOLDER, "no_face"), exist_ok=True)
    os.makedirs(STATIC_FOLDER, exist_ok=True)

    AppPaths.APP_DIR=APP_DIR
    AppPaths.STATIC_FOLDER=STATIC_FOLDER
    AppPaths.UPLOAD_FOLDER=UPLOAD_FOLDER

    model_loader:ModelLoader=container.default_model_loader();
    if model_loader is not None:
        for model in model_loader.model_registry["detectors"]:
            os.makedirs(os.path.join(STATIC_FOLDER, model), exist_ok=True)

    image_storage.init_storage(model_loader=model_loader);