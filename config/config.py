from pydantic import Field
from pydantic_settings import BaseSettings,SettingsConfigDict
from enum import Enum
from typing import Optional

class ProcessingType(str, Enum):
    Local = "local"
    Triton = "triton"
    Mixed = "mixed"

class StoreType(str, Enum):
    Memory = "memory"
    Milvus = "milvus"

class StoreSettings(BaseSettings):
    Type: StoreType = Field(StoreType.Memory)
    URL: str = Field("http://localhost:19530")

    class Config:
        env_prefix = 'Store_'


class ProcessingSettings(BaseSettings):
    Type: ProcessingType = Field(ProcessingType.Local)
    TritonURL: Optional[str] = Field("http://localhost:8081")

    class Config:
        env_prefix = 'Processing_'

class Settings(BaseSettings):
    Store: StoreSettings = StoreSettings()
    Processing: ProcessingType = Field(ProcessingType.Local)

    class Config:
        env_file = '.env'
        env_file_encoding = 'utf-8'
        case_sensitive = False
