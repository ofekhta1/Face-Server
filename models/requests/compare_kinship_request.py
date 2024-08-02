from .compare_faces_request import CompareFacesRequest
from typing import List,Annotated,Optional
from fastapi import  Body
from models.embedder_name import EmbedderName

class CompareKinshipRequest(CompareFacesRequest):
    kinship_embedder_name:Annotated[EmbedderName,Body()]=EmbedderName.bb
    similarity_threshold:Annotated[Optional[float],Body(validation_alias="similarity_thresh",alias="similarity_thresh")]=0.5
    quality_threshold:Annotated[Optional[float],Body(alias="quality_thresh",validation_alias="quality_thresh")]=0
