from Shared.models.embedder_name import EmbedderName
from .base_response import BaseResponse

class CompareKinshipResponse(BaseResponse):
    average_similarities:dict[EmbedderName,float]
    cluster1_count:int
    cluster2_count:int

