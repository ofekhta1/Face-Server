import os
from pydantic import Field,BaseModel
from pydantic_settings import BaseSettings
from enum import Enum
from typing import Optional

class ProcessingType(str, Enum):
    Local = "local"
    Triton = "triton"
    Mixed = "mixed"

class StoreType(str, Enum):
    Memory = "memory"
    Milvus = "milvus"

class QueueType(str, Enum):
    Memory = "memory"
    RabbitMQ = "rabbitmq"

class StoreSettings(BaseModel):
    Type: StoreType = Field(StoreType.Memory)
    URL: str = Field("http://localhost:19530")

class QueueSettings(BaseModel):
    Type: QueueType = Field(QueueType.Memory)
    URL: str = Field("amqp://localhost:5672")
    Consume:bool=Field(True)

class ProcessingSettings(BaseModel):
    Type: ProcessingType = Field(ProcessingType.Local)
    TritonURL: Optional[str] = Field("localhost:8080")


class Settings(BaseSettings):
    Store: StoreSettings
    Processing: ProcessingSettings 
    Queue: QueueSettings 
    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        env_nested_delimiter="__"
        case_sensitive = True

