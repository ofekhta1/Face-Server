from models.responses import UploadImagesResponse
from services import util
from services.processing.image_loader import ImageLoader
from config.app_paths import AppPaths
import os
from models.errors.base_error import BaseError
import shutil
import traceback
from fastapi.encoders import jsonable_encoder
from models.processing_message import ProcessingMessage
from fastapi.responses import JSONResponse
import sys
import numpy as np
from fastapi import APIRouter,UploadFile,File,Body,Depends,HTTPException
from routes.resources import Container
from typing import Annotated,Optional,List
from models import DetectorName,EmbedderName
from dependency_injector.wiring import inject, Provide
from services.processing.local.local_image_processor import LocalImageProcessor
from services.processing.triton.triton_image_processor import TritonImageProcessor
from services.stores import InMemoryImageEmbeddingManager,MilvusImageEmbeddingManager
from services.queue import InMemoryProcessingQueue
import zipfile

# Define directories
APP_DIR = os.path.dirname(sys.argv[0])
UPLOAD_FOLDER = os.path.join(APP_DIR, "pool")
STATIC_FOLDER = os.path.join(APP_DIR, "static")

# create dirs
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER, "no_face"), exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

AppPaths.APP_DIR=APP_DIR
AppPaths.STATIC_FOLDER=STATIC_FOLDER
AppPaths.UPLOAD_FOLDER=UPLOAD_FOLDER

file_handling_router=APIRouter()


@file_handling_router.post("/api/upload")
@inject
async def upload_image(
    files: Annotated[List[UploadFile], File()],
    return_detector: Annotated[DetectorName, Body(alias="detector_name")]= DetectorName.retinaface_buffalo,
    return_embedder: Annotated[EmbedderName, Body(alias="embedder_name")]= EmbedderName.resnet100,
    save_invalid: Annotated[Optional[bool], Body()]=False,
    image_processor:LocalImageProcessor|TritonImageProcessor=Depends(Provide[Container.default_image_processor]),
    emb_manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager=Depends(Provide[Container.emb_manager]),)->UploadImagesResponse:
    file_names=[]
    errors=[]
    for file in files:
        if file.filename:
            filename = file.filename.replace("_", "")
            if ImageLoader.allowed_file(file.filename):
                path = os.path.join(AppPaths.UPLOAD_FOLDER, file.filename)
                try:
                    # save image
                    with open(path,"wb") as buffer:
                        shutil.copyfileobj(file.file,buffer)
                        file_names.append(file.filename)

                except Exception as e:
                    tb = traceback.format_exc()
                    errors.append(f"Failed to save {filename} due to error: {str(e)}")

                finally:
                    file.file.close()
                    
    response = await internal_image_upload(file_names,emb_manager,image_processor,return_detector,return_embedder,save_invalid)
    if isinstance(response,BaseError):
        raise HTTPException(500,jsonable_encoder(response));
    response.errors=response.errors+errors
    return response;


async def internal_image_upload(file_names,emb_manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager,
                                image_processor:LocalImageProcessor|TritonImageProcessor,
                                return_detector:DetectorName,return_embedder:EmbedderName,save_invalid:bool)->UploadImagesResponse|BaseError:

    errors:list = []
    faces_length = []
    detector_indices: list[dict[str, list[int]]] = [{}]*len(file_names)
    valid_images = []
    generated_embeddings: list[dict[str, list[np.ndarray]]] = {}
    # get request parameters    gender_age_model = ModelLoader.load_genderage("MobileNetCeleb0.25_CelebA")


    # true if images will be saved without containing faces
    i = -1
    invalid_images = []
    for file in file_names:
        i += 1
        path = os.path.join(AppPaths.UPLOAD_FOLDER, file)
        # load model
        generated_embeddings,errors=await image_processor.process_image(file)
        return_key=f"{return_detector}_{return_embedder}"
        if return_key not in generated_embeddings or len(generated_embeddings[return_key])==0:
            faces_length.append(0)
            # if images with no detected faces are allowed save them under the no face directory
            if save_invalid:
                os.replace(
                    path,
                    os.path.join(
                        AppPaths.UPLOAD_FOLDER, "no_face", file
                    ),
                )
            else:
                os.remove(path)
            invalid_images.append("no_face/" + file)
        else:
            valid_images.append(file)
            embeddings=generated_embeddings[return_key]
            faces_length.append(len(embeddings))
        # save the current database state
        emb_manager.save()
        indices_result= util.get_all_detectors_faces(
            generated_embeddings, return_detector,image_processor.model_loader
        )
        if isinstance(indices_result,BaseError):
            return indices_result
        detector_indices[i] =indices_result

    response = UploadImagesResponse(images=valid_images,invalid_images=invalid_images,
                                detector_indices=detector_indices,
                                faces_length=faces_length,errors=errors);
    return response;


@file_handling_router.post("/api/upload_zip")
@inject
async def upload_zip(
    file: Annotated[UploadFile, File()],
    return_detector: Annotated[DetectorName, Body(alias="detector_name")]= DetectorName.retinaface_antelope,
    return_embedder: Annotated[EmbedderName, Body(alias="embedder_name")]= EmbedderName.resnet100,
    save_invalid: Annotated[Optional[bool], Body()]=False,
    
    queue:InMemoryProcessingQueue=Depends(Provide[Container.queue]),
    image_processor:LocalImageProcessor|TritonImageProcessor=Depends(Provide[Container.default_image_processor]),
    emb_manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager=Depends(Provide[Container.emb_manager]),
    )->UploadImagesResponse:
    file_names=[]
    errors=[]
    if not file.filename.endswith('.zip'):
        return JSONResponse(status_code=400, content={"message": "Invalid file type. Only .zip files are allowed."})
    zip_path = os.path.join(AppPaths.UPLOAD_FOLDER, file.filename)
    with open(zip_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
    try:
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            for member in zip_ref.namelist():
                if ImageLoader.allowed_file(member):
                    safe_filename=member.replace("/",'')
                    dest_path=os.path.join(AppPaths.UPLOAD_FOLDER,safe_filename)
                    with zip_ref.open(member) as source,open(dest_path,"wb") as target:
                        shutil.copyfileobj(source, target)
                        file_names.append(safe_filename)
    except zipfile.BadZipFile:
        os.remove(zip_path)
        return JSONResponse(status_code=400, content={"message": "Failed to unzip the file. It may be corrupted."})

    # Remove the zip file after extraction
    os.remove(zip_path)
    message=ProcessingMessage(image_paths=file_names,return_detector=return_detector,return_embedder=return_embedder, save_invalid=save_invalid)
    queue.enqueue(message)
    return UploadImagesResponse(images=file_names,invalid_images=[],detector_indices=[],faces_length=[]);
    # response = await internal_image_upload(file_names,emb_manager,image_processor,return_detector,return_embedder,save_invalid)
    
    # if isinstance(response,BaseError):
        # raise HTTPException(500,jsonable_encoder(response));
    # response.errors=response.errors+errors
    # return response



@file_handling_router.get("/api/gallery")
@inject
def get_gallery(embedder_name:EmbedderName,detector_name:DetectorName,
    emb_manager:InMemoryImageEmbeddingManager|MilvusImageEmbeddingManager=Depends(Provide[Container.emb_manager]))->list[str]:

    embeddings=emb_manager.get_all_embeddings(detector_name,embedder_name,False)
    result= [e.name for e in embeddings]
    return result
