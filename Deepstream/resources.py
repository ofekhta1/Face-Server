from Shared.services.models import TritonModelLoader
from Shared.services.stores.metadata_manager import MetadataManager
from Shared.services.stores.embeddings import MilvusImageEmbeddingManager

from Shared.services.processing.face_aligner import FaceAligner
from Shared.services.processing.triton.triton_image_processor import (
    TritonImageProcessor,
)
from services.frame_extractor import FrameExtractor
from services.deepstream_pipeline import DeepstreamPipeline
from Shared.services.processing.triton.triton_face_extractor import TritonFaceExtractor
from Shared.services.stores.media import MinioImageStorage
from Shared.services.processing.triton.triton_embedding_generator import (
    TritonEmbeddingGenerator,
)
from Shared.services.stores.media import LocalImageStorage,MinioImageStorage,LocalVideoStorage,MinioVideoStorage,BaseMediaStorage
from config.settings import Settings
from Shared.services.logging import ConsoleLogger
from dependency_injector import containers, providers
from Shared.services.queue import RabbitMQQueue
from deepstream_consumer import DeepstreamConsumer
from Shared.services.processing.triton.triton_client_handler import TritonClientHandler
from Shared.services.stores.jobs import BaseJobManager,InMemoryJobManager,RedisJobManager
from services.data_uploader import DataUploader

class Container(containers.DeclarativeContainer):
    config: Settings = providers.Configuration()
    logger = providers.Singleton(ConsoleLogger, config.Logging.Verbose)
    default_model_loader = providers.Singleton(TritonModelLoader)

    image_storage = providers.Singleton(
        MinioImageStorage,
        endpoint=config.Store.Media.URL,
        access_key=config.Store.Media.AccessKey,
        secret_key=config.Store.Media.SecretKey,
        logger=logger,
    )

    emb_manager = providers.Singleton(
        MilvusImageEmbeddingManager,
        url=config.Store.Embedding.URL,
        model_loader=default_model_loader,
        logger=logger
    )

    face_extractor = providers.Singleton(TritonFaceExtractor, default_model_loader)

    embedding_generator = providers.Singleton(
            TritonEmbeddingGenerator, emb_manager, face_extractor, image_storage
        )

    face_aligner = providers.Singleton(FaceAligner, face_extractor, image_storage)

    metadata_manager = providers.Singleton(
        MetadataManager,
        emb_manager,
        face_aligner,
        embedding_generator,
        default_model_loader,
    )
    queue = providers.Singleton(RabbitMQQueue, config.Queue.Host, config.Queue.Port,"face_video_queue")
    
    frame_extractor = providers.Singleton(FrameExtractor,logger)

    data_uploader = providers.Singleton(DataUploader,frame_extractor,image_storage,
                                        emb_manager,default_model_loader,config.Processing.Postprocess,logger)
    
    job_manager=providers.Selector(
        config.Store.Job.Type,
        memory=providers.Singleton(InMemoryJobManager,logger),
        redis=providers.Singleton(RedisJobManager,config.Store.Job.Host,config.Store.Job.Port,logger)
    )
    pipeline=providers.Singleton(
        DeepstreamPipeline,config.Processing.Output,data_uploader,logger
    )
    video_storage=providers.Selector(config.Store.Media.Type,
                                local=providers.Singleton(LocalVideoStorage,logger=logger),
                                minio=providers.Singleton(MinioVideoStorage, endpoint=config.Store.Media.URL,
                                                        access_key=config.Store.Media.AccessKey,
                                                        secret_key=config.Store.Media.SecretKey,
                                                        logger=logger))
    consumer = providers.Singleton(
        DeepstreamConsumer,
        queue,
        job_manager,
        image_storage,
        video_storage,
        pipeline,
        f"{config.Queue.Host}:{config.Queue.Port}",
    )
    
    face_extractor = providers.Singleton(TritonFaceExtractor, default_model_loader)


async def init_resources():

    container = Container()
    settings = Settings()

    container.config.from_dict(settings.model_dump())
    container.init_resources()

    TritonClientHandler.init(container.config.Processing.TritonURL())

    return container
