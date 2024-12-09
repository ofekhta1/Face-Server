from typing import Optional, Annotated
from pydantic import BaseModel
from Shared.models.detector_name import DetectorName
from Shared.models.embedder_name import EmbedderName
from uuid import UUID

class ProcessingMessage(BaseModel):
    id:UUID
    data_paths: list[str] = []
    return_detector: DetectorName = DetectorName.retinaface_buffalo
    return_embedder: EmbedderName = EmbedderName.resnet100
    save_invalid: bool = False
