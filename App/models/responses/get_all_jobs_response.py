from uuid import UUID
from Shared.models.job_metadata import JobMetadata
from .base_response import BaseResponse
class GetAllJobsResponse(BaseResponse):
    jobs:dict[UUID,JobMetadata]
