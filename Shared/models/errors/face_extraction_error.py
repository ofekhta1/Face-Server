from Shared.models.detector_name import DetectorName
from .base_error import BaseError
class FaceExtractionError(BaseError):
    detector_name:str|DetectorName