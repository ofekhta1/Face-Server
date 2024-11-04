
from fastapi import APIRouter,HTTPException,Depends,Response
from routes.resources import Container
from dependency_injector.wiring import inject, Provide
from services.processing.face_clustering import FaceClustering
from services.queue import RabbitMQQueue,InMemoryProcessingQueue
from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
from prometheus_client import Gauge, Histogram, Counter

metrics_router=APIRouter()


@metrics_router.get("/metrics")
@inject
def metrics(queue:RabbitMQQueue|InMemoryProcessingQueue=Depends(Provide[Container.queue]))-> bytes:
    
    queue.update()

    return Response(content=generate_latest(),media_type=CONTENT_TYPE_LATEST)
   
@metrics_router.get("/health")
def health_check():
    return {"status": "healthy"}