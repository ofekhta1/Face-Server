from .base_request import BaseRequest
from typing import List,Annotated
from fastapi import   Form


class CompareKinshipClustersRequest(BaseRequest):
    cluster_id_1:str
    cluster_id_2:str