import redis.asyncio.client
from Shared.models.job_type import JobType
from Shared.models.job_status import JobStatus
from Shared.models.job_metadata import JobMetadata
from uuid import UUID
from typing import List
from Shared.services.logging import ConsoleLogger
from Shared.models.errors.base_error import BaseError
import redis
import datetime
import json
import asyncio
from .base_job_manager import BaseJobManager


class RedisJobManager(BaseJobManager):
    def __init__(self, host: str, port: int, logger:ConsoleLogger):
        self.client = redis.asyncio.client.Redis(host=host, port=port, db=1,decode_responses=True)
        self.logger = logger

    def _get_job_key(self, job_id):
        return f"jobs:{job_id}"
    
    async def create_job(self, job_id: UUID,job_type:JobType):
        job_key = self._get_job_key(job_id)
        
        metadata: JobMetadata=JobMetadata(status=JobStatus.queued,started_at=datetime.datetime.now(datetime.timezone.utc),type=job_type)
        # Convert metadata to a dictionary
        metadata_dict = metadata.to_dict()
        metadata_dict["valid_images"]=json.dumps(metadata_dict["valid_images"])
        metadata_dict["invalid_images"]=json.dumps(metadata_dict["invalid_images"])
        # Use HSET to store job metadata in Redis
        await self.client.hset(job_key, mapping=metadata_dict)
        self.logger.info(f"Job {job_id} created with metadata: {metadata}")
        
    async def update_job_status(self, job_id: UUID, job_status: JobStatus)->bool|BaseError:
        job_key = self._get_job_key(job_id)
        if await self.client.exists(job_key):
            await self.client.hset(job_key, "status", job_status.value)
            self.logger.info(f"Job {job_id} updated to status: {job_status}")
            return True
        else:
            self.logger.error(f"Job {job_id} not found")
            return BaseError(reason=f"Job {job_id} not found")
        
    async def get_job_status(self, job_id: UUID)->JobStatus|BaseError:
        job_key = self._get_job_key(job_id)
        if await self.client.exists(job_key):
            status_str =await self.client.hget(job_key, "status")
            if status_str:
                return JobStatus(status_str)
            else:
                return BaseError(reason=f"Job {job_id} status not found")
        else:
            self.logger.error(f"Job {job_id} not found")
            return BaseError(reason=f"Job {job_id} not found")
    
    async def get_job_details(self, job_id: UUID)->JobMetadata|BaseError:
        job_key=self._get_job_key(job_id)
        if await self.client.exists(job_key):
            metadata=await self.client.hgetall(job_key)
            if metadata:
                metadata_dict={k: v for k, v in metadata.items()}
                metadata_dict["valid_images"]=json.loads(metadata_dict["valid_images"])
                metadata_dict["invalid_images"]=json.loads(metadata_dict["invalid_images"])
                return JobMetadata.from_dict(metadata_dict)
            else:
                return BaseError(reason=f"Job {job_id} metadata not found")
        else:
            self.logger.error(f"Job {job_id} not found")
            return BaseError(reason=f"Job {job_id} not found")
        


    async def delete_job(self, job_id: UUID)->bool:
        job_key = self._get_job_key(job_id)
        if await self.client.exists(job_key):
            await self.client.delete(job_key)
            self.logger.info(f"Job {job_id} deleted")
            return True
        else:
            self.logger.error(f"Job {job_id} not found")
            return False  # Job not found, so we don't raise an error here. Instead, we return False.
        
    async def get_all_jobs(self) -> List[tuple[UUID, JobMetadata]]:
        pipeline = self.client.pipeline()
        job_keys = await self.client.keys("jobs:*")

        # Prepare pipeline for batch retrieval
        for job_key in job_keys:
            pipeline.hgetall(job_key)

        # Execute pipeline
        results = await pipeline.execute()

        jobs = []
        for job_key, job_data in zip(job_keys, results):
            if job_data:
                job_id = UUID(job_key.split(":")[1])
                metadata = JobMetadata.from_dict({k: v for k, v in job_data.items()})
                jobs.append((job_id, metadata))

        return jobs
    
    async def complete_job(self, job_id: UUID, valid_images: List[str], invalid_images: List[str]):
        job_key = self._get_job_key(job_id)
        try:
            # Create a pipeline
            pipeline = self.client.pipeline()
            
            pipeline.multi()
            
            # Queue the operations individually
            pipeline.hset(job_key, "valid_images", json.dumps(valid_images))
            pipeline.hset(job_key, "invalid_images", json.dumps(invalid_images))
            pipeline.hset(job_key, "status", JobStatus.completed.value)
            pipeline.hset(job_key,"completed_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
            # Execute all queued operations in one go
            await pipeline.execute()
            
            self.logger.info(f"Job {job_id} completed with valid images: {valid_images}, invalid images: {invalid_images}")
        except Exception as e:
            self.logger.error(f"Failed to complete job {job_id}: {str(e)}")
            raise e  # re-raise the exception for caller to handle

    async def fail_job(self, job_id: UUID, error_message: str,invalid_images:list[str],valid_images:list[str]=[]):
        job_key = self._get_job_key(job_id)
        pipeline = self.client.pipeline()

        pipeline.multi()
        pipeline.hset(job_key, "error_message", error_message)
        pipeline.hset( job_key, "status", JobStatus.failed.value)
        pipeline.hset(job_key,"completed_at", datetime.datetime.now(datetime.timezone.utc).isoformat())
        pipeline.hset(job_key, "valid_images", json.dumps(valid_images))
        pipeline.hset(job_key, "invalid_images", json.dumps(invalid_images))
        await pipeline.execute()
        self.logger.error(f"Job {job_id} failed with error: {error_message}")