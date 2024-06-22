from modules import (
    InMemoryImageEmbeddingManager,
    MilvusImageEmbeddingManager,
    ImageHelper,
    ImageGroupRepository,
    ModelLoader,
    config,
    AppPaths
)
from models.stored_embedding import StoredEmbeddings
import sys
if "resources" not in sys.modules:
    helper=None;
    manager=None;
    groups=None;
    cfg=None;

def init_resources():
    global groups,helper,manager,cfg
    cfg = config.load_config("config.json")
    match cfg.store.lower():
        case "milvus":
            manager = MilvusImageEmbeddingManager("http://localhost:19530")
        case "memory":
            manager = InMemoryImageEmbeddingManager(AppPaths.APP_DIR)
        case _:
            manager = InMemoryImageEmbeddingManager(AppPaths.APP_DIR)

    groups = ImageGroupRepository(AppPaths.APP_DIR)
    helper = ImageHelper(groups, manager, AppPaths.UPLOAD_FOLDER,AppPaths.STATIC_FOLDER)
    for model_name, _ in ModelLoader.detectors.items():
        ModelLoader.load_detector(model_name, AppPaths.APP_DIR)
        manager.load(model_name)
        for embedder_name, _ in ModelLoader.embedders.items():
            if embedder_name not in manager.db_embeddings[model_name].embeddings:
                manager.db_embeddings[model_name].embeddings[embedder_name] = (
                    StoredEmbeddings([])
                )