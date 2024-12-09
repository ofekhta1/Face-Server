from App.models.responses import UploadImagesResponse,UploadBatchResponse,UploadVideoResponse
from Shared.models.job_type import JobType
from Shared.services import util
from Shared.services.processing.media_loader import MediaLoader
import os
from Shared.models.errors.base_error import BaseError
import shutil
from Shared.services.util import face_path
import traceback
import tempfile
from fastapi.encoders import jsonable_encoder
from Shared.models.processing_message import ProcessingMessage
from fastapi.responses import JSONResponse
from Shared.models.face_info import FaceInfo
import numpy as np
from fastapi import APIRouter,UploadFile,File,Body,Depends,HTTPException
from routes.resources import Container
from typing import Annotated,Optional,List
from Shared.models import DetectorName,EmbedderName
from dependency_injector.wiring import inject, Provide
from Shared.services.processing.local.local_image_processor import LocalImageProcessor
from Shared.services.processing.triton.triton_image_processor import TritonImageProcessor
from Shared.services.stores.media import BaseImageStorage,BaseVideoStorage
from Shared.services.stores.jobs import BaseJobManager
from Shared.services.stores.embeddings import BaseImageEmbeddingManager
from Shared.services.stores.metadata_manager import MetadataManager
from Shared.services.queue import InMemoryProcessingQueue
import zipfile
from fastapi.responses import StreamingResponse
import os
import mimetypes
import uuid

file_handling_router=APIRouter()



@file_handling_router.get("/static/{file_path:path}")
@inject

async def stream_static(file_path: str,
                        image_storage:BaseImageStorage=Depends(Provide[Container.image_storage])):
    file_stream=image_storage.serve_media_file(file_path,True);

    if file_stream is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    media_type, _ = mimetypes.guess_type(file_path)
    media_type = media_type or "application/octet-stream"

    # Use a context manager to open the file
    return StreamingResponse(file_stream, media_type=media_type)
    
@file_handling_router.get("/pool/request{file_path:path}")
@inject
async def stream_pool(file_path: str,
                        image_storage:BaseImageStorage=Depends(Provide[Container.image_storage])):
                      
    file_stream=image_storage.serve_media_file(file_path,False);

    if file_stream is None:
        raise HTTPException(status_code=404, detail="File not found")
    
    media_type, _ = mimetypes.guess_type(file_path)
    media_type = media_type or "application/octet-stream"

    # Use a context manager to open the file
    return StreamingResponse(file_stream, media_type=media_type)
    
@file_handling_router.post("/api/files/upload")
@inject
async def upload_image(
    files: Annotated[List[UploadFile], File()],
    return_detector: Annotated[DetectorName, Body(alias="detector_name")]= DetectorName.retinaface_buffalo,
    return_embedder: Annotated[EmbedderName, Body(alias="embedder_name")]= EmbedderName.resnet100,
    save_invalid: Annotated[Optional[bool], Body()]=False,
    image_storage:BaseImageStorage=Depends(Provide[Container.image_storage]),
    image_processor:LocalImageProcessor|TritonImageProcessor=Depends(Provide[Container.default_image_processor]),
    image_metadata:MetadataManager=Depends(Provide[Container.metadata_manager]),
    emb_manager:BaseImageEmbeddingManager=Depends(Provide[Container.emb_manager]),)->UploadImagesResponse:
    file_names=[]
    errors=[]
    for file in files:
        if file.filename:
            filename = file.filename.replace("_", "")
            try:
                _, file_extension = os.path.splitext(filename)
                with tempfile.NamedTemporaryFile(delete=False,suffix=file_extension) as temp_file:
                # Write the uploaded file to the temporary file
                    shutil.copyfileobj(file.file, temp_file)
                
                file_names.append((temp_file.name,file.filename))

            except Exception as e:
                tb = traceback.format_exc()
                errors.append(f"Failed to save {filename} due to error: {str(e)}")

            finally:
                file.file.close()
                
    response = await internal_image_upload(file_names,emb_manager,image_processor,image_storage,image_metadata,return_detector,return_embedder,save_invalid)
    for tempfile_name,_ in file_names:
        try:
            os.remove(filename)
        except OSError:
            pass
        shutil.rmtree(tempfile_name+"_faces")
        
    if isinstance(response,BaseError):
        raise HTTPException(500,jsonable_encoder(response));
    response.errors=response.errors+errors
    return response;


async def internal_image_upload(file_names:tuple[str,str],emb_manager:BaseImageEmbeddingManager,
                                image_processor:LocalImageProcessor|TritonImageProcessor,
                                image_storage:BaseImageStorage,
                                image_metadata:MetadataManager,
                                return_detector:DetectorName,return_embedder:EmbedderName,save_invalid:bool)->UploadImagesResponse|BaseError:

    errors:list = []
    faces_length = []
    detector_indices: list[dict[str, list[int]]] = [{}]*len(file_names)
    detector_metadata: list[dict[str, list[FaceInfo]]] = [{}]*len(file_names)
    
    valid_images = []
    generated_embeddings: list[dict[str, list[np.ndarray]]] = {}
    # get request parameters    gender_age_model = ModelLoader.load_genderage("MobileNetCeleb0.25_CelebA")


    # true if images will be saved without containing faces
    i = -1
    invalid_images = []
    for temp_file_path,filename in file_names:
        i += 1
        temp_dir = os.path.dirname(temp_file_path)
        faces_dir=os.path.join(temp_dir,os.path.basename(temp_file_path) + '_faces')
        os.makedirs(faces_dir,exist_ok=True);

        # load model
        generated_embeddings,detector_metadata[i],errors=await image_processor.process_image(temp_file_path,filename,faces_dir)
        return_key=f"{return_detector}_{return_embedder}"
        if return_key not in generated_embeddings or len(generated_embeddings[return_key])==0:
            faces_length.append(0)
            # if images with no detected faces are allowed save them under the no face directory
            if save_invalid:
                image_storage.fsave_media_file(temp_file_path,filename,invalid=True)

            os.remove(temp_file_path)
            invalid_images.append("no_face/" + filename)
            continue;
        else:
            valid_images.append(filename)
            embeddings=generated_embeddings[return_key]
            faces_length.append(len(embeddings))
        
        image_storage.fsave_media_file(temp_file_path,filename)
        seen_detectors=[];
        for models in generated_embeddings:
            detector,_=models.split('_')
            if detector not in seen_detectors:
                seen_detectors.append(detector)
                count=len(generated_embeddings[models])
                for face_num in range(count):
                    aligned_filename=face_path(filename,face_num)
                    path=os.path.join(faces_dir,detector,aligned_filename)
                    image_storage.fsave_media_file(path,aligned_filename,detector);

        # save the current database state
        emb_manager.save()
        indices_result= util.get_all_detectors_faces(
            filename,generated_embeddings, return_detector,image_processor.model_loader,image_metadata
        )

        if isinstance(indices_result,BaseError):
            return indices_result
        detector_indices[i],_ =indices_result
   
    response = UploadImagesResponse(images=valid_images,invalid_images=invalid_images,
                                detector_indices=detector_indices,
                                metadata=detector_metadata,
                                faces_length=faces_length,errors=errors);
    return response;


@file_handling_router.post("/api/files/upload_zip")
@inject
async def upload_zip(
    file: Annotated[UploadFile, File()],
    return_detector: Annotated[DetectorName, Body(alias="detector_name")]= DetectorName.retinaface_antelope,
    return_embedder: Annotated[EmbedderName, Body(alias="embedder_name")]= EmbedderName.resnet100,
    save_invalid: Annotated[Optional[bool], Body()]=False,
    image_storage:BaseImageStorage=Depends(Provide[Container.image_storage]),
    job_manager:BaseJobManager=Depends(Provide[Container.job_manager]),
    queue:InMemoryProcessingQueue=Depends(Provide[Container.image_queue]),
    )->UploadBatchResponse:
    file_names=[]
    invalid_file_names=[]
    if not file.filename.endswith('.zip'):
        return JSONResponse(status_code=400, content={"message": "Invalid file type. Only .zip files are allowed."})
    with tempfile.NamedTemporaryFile(delete=False) as temp_zip_file:
        try:
            shutil.copyfileobj(file.file, temp_zip_file)
            zip_path=temp_zip_file.name
            with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                for member in zip_ref.infolist():
                    if MediaLoader.allowed_img_file(member.filename):
                        safe_filename=member.filename.replace("/",'')
                        with zip_ref.open(member) as source:
                            success=image_storage.save_media_file(source,safe_filename,member.file_size);
                            if success:
                                file_names.append(safe_filename)
                            else:
                                return JSONResponse(status_code=400, content={"message": "Failed to upload the images to storage"})

                    else:
                        invalid_file_names.append(member.filename)

        except zipfile.BadZipFile as ex:
            os.remove(zip_path)
            return JSONResponse(status_code=400, content={"message": "Failed to unzip the file. It may be corrupted."})
        
    # Remove the zip file after extraction
    os.remove(zip_path)
    if len(file_names)==0:
        return JSONResponse(status_code=400, content={"message": "No valid images found in the uploaded zip file."})
    job_id=uuid.uuid4()
    message=ProcessingMessage(id=job_id,data_paths=file_names,return_detector=return_detector,return_embedder=return_embedder, save_invalid=save_invalid)
    await job_manager.create_job(job_id,JobType.image)
    if queue.enqueue(message):
        return UploadBatchResponse(job_id=job_id,images=file_names,invalid_images=invalid_file_names);
    else:
        await job_manager.delete_job(job_id)
        return JSONResponse(status_code=400, content={"message": "Failed to enqueue the processing job"})



@file_handling_router.get("/api/files/gallery")
@inject
def get_gallery(embedder_name:EmbedderName,detector_name:DetectorName,
    emb_manager:BaseImageEmbeddingManager=Depends(Provide[Container.emb_manager]))->list[str]:

    embeddings=emb_manager.get_all_embeddings(detector_name,embedder_name,True)
    result= [e.name for e in embeddings]
    return result

@file_handling_router.post("/api/files/upload_video")
@inject
async def upload_video(
    file: Annotated[UploadFile, File()],
    return_detector: Annotated[DetectorName, Body(alias="detector_name")]= DetectorName.retinaface_buffalo,
    return_embedder: Annotated[EmbedderName, Body(alias="embedder_name")]= EmbedderName.resnet100,
    save_invalid: Annotated[Optional[bool], Body()]=False,
    job_manager:BaseJobManager=Depends(Provide[Container.job_manager]),
    queue:InMemoryProcessingQueue=Depends(Provide[Container.video_queue]),
    video_storage:BaseVideoStorage=Depends(Provide[Container.video_storage])):
    
    if not video_storage.allowed_file(file.filename):
        return JSONResponse(status_code=400, content={"message": f"Invalid file type. Allowed types are {MediaLoader.ALLOWED_VID_EXTENSIONS}"})

    try:
        safe_filename=file.filename.replace("/",'')
        success=video_storage.save_media_file(file.file,safe_filename,file.size,return_detector)
        if not success:
            return JSONResponse(status_code=400, content={"message": "Failed to upload the images to storage"})
    except Exception as ex:
        return JSONResponse(status_code=400, content={"message": "Failed to upload the images to storage"})
        
    job_id=uuid.uuid4()
    message=ProcessingMessage(id=job_id,data_paths=[safe_filename],return_detector=return_detector,return_embedder=return_embedder, save_invalid=save_invalid)
    await job_manager.create_job(job_id,JobType.video)
    if queue.enqueue(message):
        return UploadVideoResponse(job_id=job_id,video_path=safe_filename);
    else:
        await job_manager.delete_job(job_id)
        return JSONResponse(status_code=400, content={"message": "Failed to enqueue the processing job"})

