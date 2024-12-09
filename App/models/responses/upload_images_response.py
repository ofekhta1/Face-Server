from typing import Annotated,Optional,List,Dict
from .base_response import BaseResponse
from Shared.models.detector_name import DetectorName
from Shared.models.face_info import FaceInfo
from uuid import UUID
class UploadImagesResponse(BaseResponse ):
    images:List[str]
    invalid_images:List[str]
    detector_indices:List[Dict[DetectorName,List[int]]]
    metadata:Optional[List[Dict[DetectorName,List[FaceInfo]]]]
    faces_length:List[int]

class UploadBatchResponse(BaseResponse ):
    job_id: UUID
    images:List[str]
    invalid_images:List[str]

class UploadVideoResponse(BaseResponse):
    job_id: UUID
    video_path: str