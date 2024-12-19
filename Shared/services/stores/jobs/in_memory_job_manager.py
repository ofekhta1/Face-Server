from Shared.models.job_type import JobType
from Shared.models.job_status import JobStatus
from Shared.models.job_metadata import JobMetadata
from uuid import UUID
from typing import List
from Shared.services.logging import ConsoleLogger
from Shared.models.errors.base_error import BaseError
from .base_job_manager import BaseJobManager
import datetime

class InMemoryJobManager(BaseJobManager):
    def __init__(self,logger:ConsoleLogger):
        self.jobs:dict[UUID,JobMetadata] = {}
        self.logger = logger

    async def create_job(self, job_id: UUID,job_type:JobType):
        metadata=JobMetadata(status=JobStatus.queued,started_at=datetime.datetime.now(datetime.timezone.utc),type=job_type)
        self.jobs[job_id] = metadata

    async def update_job_status(self, job_id: UUID, job_status: JobStatus)->bool|BaseError:
        if job_id in self.jobs:
            self.jobs[job_id].status = job_status
            self.logger.info(f"Job {job_id} updated to status: {job_status}")
            return True
        else:
            self.logger.error(f"Job {job_id} not found")
            return BaseError(f"job {job_id} not found")

    async def get_job_status(self, job_id: UUID) -> JobStatus|BaseError:
        if job_id in self.jobs:
            return self.jobs[job_id].status
        else:
            return BaseError(f"Job {job_id} not found")

    async def get_job_details(self, job_id: UUID) -> JobMetadata|BaseError:
        if job_id in self.jobs:
            return self.jobs[job_id]
        else:
            return BaseError(f"Job {job_id} not found")

    async def delete_job(self, job_id: UUID):
        if job_id in self.jobs:
            del self.jobs[job_id]
            self.logger.info(f"Job {job_id} deleted")
            return True
        else:
            self.logger.error(f"Job {job_id} not found")
            return False  # Job not found, so we don't raise an error here. Instead, we return False.

    async def get_all_jobs(self) -> List[tuple[UUID, JobMetadata]]:
        return [(job_id, metadata) for job_id, metadata in self.jobs.items()]
    
    async def complete_job(self, job_id, valid_images, invalid_images):
        if job_id in self.jobs:
            self.jobs[job_id].completed_at = datetime.datetime.now(datetime.timezone.utc)
            self.jobs[job_id].valid_images = valid_images
            self.jobs[job_id].invalid_images = invalid_images
            self.jobs[job_id].status = JobStatus.completed
            self.logger.info(f"Job {job_id} completed with valid images: {valid_images}, invalid images: {invalid_images}")

    async def fail_job(self, job_id, error_message, invalid_images, valid_images = ...):
        if job_id in self.jobs:
            self.jobs[job_id].completed_at = datetime.datetime.now(datetime.timezone.utc)
            self.jobs[job_id].valid_images = valid_images
            self.jobs[job_id].invalid_images = invalid_images
            self.jobs[job_id].error_message = error_message
            self.jobs[job_id].status = JobStatus.failed
            self.logger.error(f"Job {job_id} failed: {error_message}, with valid images: {valid_images}, invalid images: {invalid_images}")