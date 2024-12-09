from Shared.services.models import ModelLoader,TritonModelLoader,LocalModelLoader
from Shared.services.stores import BaseClusterRepository,RedisClusterRepository,InMemoryClusterRepository,MetadataManager
from Shared.services.stores.embeddings import InMemoryImageEmbeddingManager,MilvusImageEmbeddingManager,BaseImageEmbeddingManager
from config.app_paths import AppPaths
from config.config import EmbeddingStoreType,MediaStoreType,ProcessingType
from Shared.services.processing.face_aligner import FaceAligner
from Shared.services.processing.face_similarity_search import FaceSimilaritySearch
from Shared.services.processing.face_clustering import FaceClustering
from Shared.services.stores.media import LocalImageStorage,MinioImageStorage,LocalVideoStorage,MinioVideoStorage,BaseMediaStorage
from config.config import Settings
from Shared.models.stored_embedding import StoredEmbeddings
from Shared.services.logging import ConsoleLogger
from dependency_injector import containers,providers
from Shared.services.queue import InMemoryProcessingQueue,Consumer,RabbitMQQueue
from Shared.services.processing.triton import TritonImageProcessor,TritonFaceExtractor,TritonEmbeddingGenerator,TritonClientHandler
from Shared.services.processing.local import LocalImageProcessor,LocalFaceExtractor,LocalEmbeddingGenerator
from Shared.services.stores.jobs import BaseJobManager,InMemoryJobManager,RedisJobManager
import os
import sys
from fastapi import FastAPI

class Container(containers.DeclarativeContainer):
    config:Settings= providers.Configuration();
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
    image_storage=providers.Selector(config.Store.Media.Type,
                                     local=providers.Singleton(LocalImageStorage,logger=logger),
                                     minio=providers.Singleton(MinioImageStorage, endpoint=config.Store.Media.URL,
                                                                access_key=config.Store.Media.AccessKey,
                                                                secret_key=config.Store.Media.SecretKey,
                                                                logger=logger))
    video_storage=providers.Selector(config.Store.Media.Type,
                                    local=providers.Singleton(LocalVideoStorage,logger=logger),
                                    minio=providers.Singleton(MinioVideoStorage, endpoint=config.Store.Media.URL,
                                                            access_key=config.Store.Media.AccessKey,
                                                            secret_key=config.Store.Media.SecretKey,
                                                            logger=logger))

    emb_manager = providers.Selector(
        config.Store.Embedding.Type,
        memory=providers.Singleton(InMemoryImageEmbeddingManager,AppPaths.APP_DIR,default_model_loader,logger=logger),
        milvus=providers.Singleton(MilvusImageEmbeddingManager, url=config.Store.Embedding.URL,model_loader=default_model_loader,logger=logger)
    )

    groups=providers.Selector(
        config.Store.Cluster.Type,
        memory=providers.Singleton(InMemoryClusterRepository,AppPaths.APP_DIR,default_model_loader,logger),
        redis=providers.Singleton(RedisClusterRepository,config.Store.Cluster.Host,config.Store.Cluster.Port,logger),
    )

    face_extractor=providers.Selector(
        config.Processing.Type,
        local=providers.Singleton(LocalFaceExtractor,default_model_loader),
        mixed= providers.Singleton(LocalFaceExtractor,default_model_loader),
        triton= providers.Singleton(TritonFaceExtractor,default_model_loader),
    )
    embedding_generator=providers.Selector(
        config.Processing.Type,
        local=providers.Singleton(LocalEmbeddingGenerator,emb_manager,face_extractor,image_storage),
        mixed=providers.Singleton(LocalEmbeddingGenerator,emb_manager,face_extractor,image_storage),
        triton=providers.Singleton(TritonEmbeddingGenerator,emb_manager,face_extractor,image_storage),
    )

    face_aligner = providers.Singleton(FaceAligner,face_extractor,image_storage)
    face_clustering=providers.Singleton(FaceClustering,emb_manager,groups)
    face_similarity_search=providers.Singleton(FaceSimilaritySearch,emb_manager,embedding_generator,image_storage,face_aligner);
    metadata_manager=providers.Singleton(MetadataManager,emb_manager,face_aligner,image_storage,embedding_generator,default_model_loader);

    job_manager=providers.Selector(
        config.Store.Job.Type,
        memory=providers.Singleton(InMemoryJobManager,logger),
        redis=providers.Singleton(RedisJobManager,config.Store.Job.Host,config.Store.Job.Port,logger)
    )
    image_queue=providers.Selector(
        config.Queue.Type,
        memory = providers.Singleton(InMemoryProcessingQueue),
        rabbitmq = providers.Singleton(RabbitMQQueue,config.Queue.Host,config.Queue.Port)
    )
    video_queue=providers.Selector(
        config.Queue.Type,
        memory = providers.Singleton(InMemoryProcessingQueue),
        rabbitmq = providers.Singleton(RabbitMQQueue,config.Queue.Host,config.Queue.Port,"face_video_queue")
    )
    default_image_processor=providers.Selector(
        config.Processing.Type,
        local= providers.Singleton(LocalImageProcessor,emb_manager,face_aligner,embedding_generator,default_model_loader),
        mixed= providers.Singleton(LocalImageProcessor,emb_manager,face_aligner,embedding_generator,default_model_loader),
        triton= providers.Singleton(TritonImageProcessor,emb_manager,face_aligner,face_extractor,default_model_loader),
    )


    consumer=providers.Selector(
        config.Queue.Type,
        memory = providers.Singleton(Consumer, emb_manager,default_image_processor,image_queue,job_manager,image_storage),
        rabbitmq = providers.Singleton(Consumer, emb_manager,default_image_processor,image_queue,job_manager,image_storage,f"{config.Queue.Host}:{config.Queue.Port}")
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
    
    manager:BaseImageEmbeddingManager=container.emb_manager();
    if settings.Queue.Consume:
        consumer=container.consumer();
        consumer.start_consuming();
    
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
    image_storage:BaseMediaStorage=container.image_storage()
    video_storage:BaseMediaStorage=container.video_storage()

    model_loader:ModelLoader=container.default_model_loader();
    if model_loader is not None:
        for model in model_loader.model_registry["detectors"]:
            os.makedirs(os.path.join(STATIC_FOLDER, model), exist_ok=True)

    image_storage.init_storage(model_loader=model_loader,pool_dir=UPLOAD_FOLDER,processed_dir=STATIC_FOLDER);
    video_storage.init_storage(model_loader=model_loader,pool_dir=UPLOAD_FOLDER,processed_dir=STATIC_FOLDER);