from Shared.models.job_status import JobStatus
from Shared.models.processing_message import ProcessingMessage
from Shared.models.job_type import JobType
from Shared.models.errors.base_error import BaseError
from App.models.requests import RetryJobRequest
from App.models.responses import GetAllJobsResponse
from Shared.services.util import face_path
from fastapi.encoders import jsonable_encoder
from fastapi import APIRouter,UploadFile,File,Body,Depends,HTTPException
from fastapi.websockets import WebSocket, WebSocketDisconnect
from routes.resources import Container
from Shared.services.queue import BaseProcessingQueue
from dependency_injector.wiring import inject, Provide
from Shared.services.stores.jobs import BaseJobManager
import os

batch_jobs_router=APIRouter()


@batch_jobs_router.get("/api/jobs/job_status")
@inject
async def get_job_status(job_id: str, job_manager: BaseJobManager=Depends(Provide[Container.job_manager]))->dict:
    result=await job_manager.get_job_status(job_id)
         
    if isinstance(result,BaseError):
        raise HTTPException(500,jsonable_encoder(result));

    return {'status':result}

@batch_jobs_router.get("/api/jobs/job_details")
@inject
async def get_job_details(job_id: str, job_manager: BaseJobManager=Depends(Provide[Container.job_manager]))->dict:
    result=await job_manager.get_job_details(job_id)
         
    if isinstance(result,BaseError):
        raise HTTPException(500,jsonable_encoder(result));

    return {'details':result}

@batch_jobs_router.get("/api/jobs")
@inject
async def get_all_jobs(job_manager: BaseJobManager=Depends(Provide[Container.job_manager]))->GetAllJobsResponse:
    result=await job_manager.get_all_jobs()
         
    if isinstance(result,BaseError):
        raise HTTPException(500,jsonable_encoder(result));
    jobs={id:details for id,details in result}
    
    return GetAllJobsResponse(jobs=jobs) 

@batch_jobs_router.post("/api/jobs/retry")
@inject
async def retry_job(request: RetryJobRequest, 
                    image_queue: BaseProcessingQueue=Depends(Provide[Container.image_queue]),
                    video_queue: BaseProcessingQueue=Depends(Provide[Container.video_queue]),
                    job_manager: BaseJobManager=Depends(Provide[Container.job_manager]))->dict:
    
    result=await job_manager.get_job_details(request.job_id)
    if isinstance(result,BaseError):
        raise HTTPException(500,jsonable_encoder(result));
    else:
        job_details=result
        message=ProcessingMessage(id=request.job_id,data_paths=job_details.invalid_images,return_detector=job_details.return_detector,
                          return_embedder=job_details.return_embedder,save_invalid=job_details.save_invalid)
        if job_details.type==JobType.image:
            image_queue.enqueue(message)
        elif job_details.type==JobType.video:
            video_queue.enqueue(message)
        else:
            raise HTTPException(400,jsonable_encoder({"error":"Unsupported job type"}));
        await job_manager.update_job_status(request.job_id,JobStatus.queued);
        return {"success":True}
    pass
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        self.active_connections.remove(websocket)

    async def broadcast(self, message: str):
        for connection in self.active_connections:
            await connection.send_text(message)

connection_manager = ConnectionManager()

@batch_jobs_router.websocket("/ws/job_updates")
async def websocket_job_updates(
    websocket: WebSocket, 
    job_manager: BaseJobManager = Depends(Provide[Container.job_manager])
):
    await connection_manager.connect(websocket)
    try:
        while True:
            # Keep the connection open
            await websocket.receive_text()
    except WebSocketDisconnect:
        connection_manager.disconnect(websocket)

