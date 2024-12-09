import os
from pydantic import Field, BaseModel
from pydantic_settings import BaseSettings
from enum import Enum
from typing import Optional

class ProcessingType(str, Enum):
    Triton = "triton"

class ImageStoreType(str, Enum):
    Minio = "minio"

class EmbeddingStoreType(str, Enum):
    Milvus = "milvus"

class QueueType(str, Enum):
    RabbitMQ = "rabbitmq"


class ClusterStoreType(str, Enum):
    Memory = "memory"
    Redis = "redis"

class JobStoreType(str, Enum):
    Memory = "memory"
    Redis = "redis"

class ClusterStoreSettings(BaseSettings):
    Type: ClusterStoreType = Field(ClusterStoreType.Memory)
    Host: str = Field("localhost")
    Port: int = Field(6379)


class EmbeddingStoreSettings(BaseSettings):
    Type: EmbeddingStoreType = Field(EmbeddingStoreType.Milvus)
    URL: str = Field("http://localhost:19530")


class JobStoreSettings(BaseSettings):
    Type: JobStoreType = Field(JobStoreType.Memory)
    Host: str = Field("localhost")
    Port: int = Field(6379)

class ImageStoreSettings(BaseSettings):
    Type: ImageStoreType = Field(ImageStoreType.Minio)
    URL: str = Field("http://localhost:9000")
    SecretKey: str = Field("MINIO_SECRET_KEY")
    AccessKey: str = Field("MINIO_ACCESS_KEY")

class StoreSettings(BaseSettings):
    Embedding: EmbeddingStoreSettings
    Cluster: ClusterStoreSettings
    Image: ImageStoreSettings
    Job: JobStoreSettings

class QueueSettings(BaseSettings):
    Type: QueueType = Field(QueueType.RabbitMQ)
    Host: str = Field("localhost")
    Port: int = Field(5672)

class ProcessingSettings(BaseSettings):
    Type: ProcessingType = Field(ProcessingType.Triton)
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