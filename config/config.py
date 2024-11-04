import os
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
from enum import Enum
from typing import Optional

class ProcessingType(str, Enum):
    Local = "local"
    Triton = "triton"
    Mixed = "mixed"

class ImageStoreType(str, Enum):
    Local = "local"
    Minio = "minio"

class EmbeddingStoreType(str, Enum):
    Memory = "memory"
    Milvus = "milvus"

class QueueType(str, Enum):
    Memory = "memory"
    RabbitMQ = "rabbitmq"

class EmbeddingStoreSettings(BaseSettings):
    Type: EmbeddingStoreType = Field(EmbeddingStoreType.Memory)
    URL: str = Field("http://localhost:19530")

class ImageStoreSettings(BaseSettings):
    Type: ImageStoreType = Field(ImageStoreType.Local)
    URL: str = Field("http://localhost:9000")
    SecretKey: str = Field("MINIO_SECRET_KEY")
    AccessKey: str = Field("MINIO_ACCESS_KEY")

class StoreSettings(BaseSettings):
    Embedding: EmbeddingStoreSettings
    Image: ImageStoreSettings

class QueueSettings(BaseSettings):
    Type: QueueType = Field(QueueType.Memory)
    Host: str = Field("localhost")
    Port: int = Field(5672)
    Consume: bool = Field(True)

class ProcessingSettings(BaseSettings):
    Type: ProcessingType = Field(ProcessingType.Local)
    TritonURL: Optional[str] = Field("localhost:8080")

class LoggingSettings(BaseSettings):
    Level: str = Field("INFO")
    Verbose: bool = Field(False)

class Settings(BaseSettings):
    Store: StoreSettings
    Processing: ProcessingSettings 
    Queue: QueueSettings
    Logging: LoggingSettings

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        env_nested_delimiter = "__"
        case_sensitive = True