from .compare_faces_request import CompareFacesRequest
from typing import List,Annotated,Optional
from fastapi import  Form
from models.embedder_name import EmbedderName

class CompareKinshipRequest(CompareFacesRequest):
    kinship_embedder_name:Annotated[EmbedderName,Form()]=EmbedderName.bb
    similarity_threshold:Annotated[float,Form(alias="similarity_thresh")]=0.5