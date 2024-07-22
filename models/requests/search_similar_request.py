from .base_request import BaseRequest
from typing import Annotated,Optional
from fastapi import  Body

class SearchMostSimilarRequest(BaseRequest):
    similarity_threshold:Annotated[Optional[float],Body(alias="similarity_thresh",validation_alias="similarity_thresh")]=0.5
    image:Annotated[str,Body()]
    selected_face:Annotated[int,Body()]=0

class SearchSimilarRequest(SearchMostSimilarRequest):
    number_of_images:Annotated[int,Body()]=5