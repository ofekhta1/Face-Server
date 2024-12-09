from pydantic import BaseModel
from uuid import UUID

class RetryJobRequest(BaseModel):
    job_id: UUID