from Shared.models.job_type import JobType
from Shared.models.job_status import JobStatus
from Shared.models.job_metadata import JobMetadata
from uuid import UUID
from typing import List
from Shared.models.errors.base_error import BaseError
class BaseJobManager:
    async def create_job(self,job_id:UUID,job_type:JobType):
        raise NotImplementedError(f"create_job not implemented in {self.__class__.__name__}");

    async def update_job_status(self, job_id: UUID, job_status:JobStatus)->bool:
        raise NotImplementedError(f"update_job not implemented in {self.__class__.__name__}");

    async def get_job_status(self, job_id: UUID) -> JobStatus:
        raise NotImplementedError(f"get_job_status not implemented in {self.__class__.__name__}");

    async def get_job_details(self, job_id: UUID) -> JobMetadata|BaseError:
        raise NotImplementedError(f"get_job_details not implemented in {self.__class__.__name__}");

    async def delete_job(self, job_id: UUID)->bool:
        raise NotImplementedError(f"delete_job not implemented in {self.__class__.__name__}");

    async def get_all_jobs(self) -> List[tuple[UUID, JobStatus]]:
        raise NotImplementedError(f"get_all_jobs not implemented in {self.__class__.__name__}");

    async def complete_job(self, job_id: UUID, valid_images:list[str],invalid_images:list[str]):
        raise NotImplementedError(f"complete_job not implemented in {self.__class__.__name__}");

    async def fail_job(self,job_id:UUID,error_message:str,invalid_images:list[str],valid_images:list[str]=[]):
        raise NotImplementedError(f"fail_job not implemented in {self.__class__.__name__}");