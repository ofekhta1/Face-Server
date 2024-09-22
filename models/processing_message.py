from typing import Optional, Annotated
from pydantic import BaseModel
from models.detector_name import DetectorName
from models.embedder_name import EmbedderName


class ProcessingMessage(BaseModel):
    image_paths: list[str] = []
    return_detector: DetectorName = DetectorName.retinaface_buffalo
    return_embedder: EmbedderName = EmbedderName.resnet100
    save_invalid: bool = False
