from models.responses import UploadImagesResponse
from modules import ModelLoader,ImageHelper,AppPaths,util
from . import resources
import os
from models.errors.base_error import BaseError
import shutil
import traceback
import sys
import numpy as np
from fastapi import APIRouter,UploadFile,File,Body,HTTPException
from typing import Annotated,Optional,List
from models import DetectorName,EmbedderName

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
async def upload_image(
    files: Annotated[List[UploadFile], File()],
    return_detector: Annotated[DetectorName, Body(alias="detector_name")]= DetectorName.retinaface_antelope,
    return_embedder: Annotated[EmbedderName, Body(alias="embedder_name")]= EmbedderName.resnet100,
    save_invalid: Annotated[Optional[bool], Body()]=False
    )->UploadImagesResponse:
    helper=resources.helper
    manager=resources.manager

    errors:list = []
    current_images = []
    faces_length = []
    detector_indices: list[dict[str, list[int]]] = []
    valid_images = []
    generated_embeddings: list[dict[str, list[np.ndarray]]] = {}
    # get request parameters

    gender_age_model = ModelLoader.load_genderage("MobileNetCeleb0.25_CelebA")
    # true if images will be saved without containing faces
    i = -1
    invalid_images = []
    for file in files:
        if file.filename:
            filename = file.filename.replace("_", "")
            if ImageHelper.allowed_file(file.filename):
                path = os.path.join(AppPaths.UPLOAD_FOLDER, file.filename)
                try:
                    i += 1
                    detector_indices.append({})
                    # save image
                    with open(path,"wb") as buffer:
                        shutil.copyfileobj(file.file,buffer)
                except Exception as e:
                    tb = traceback.format_exc()
                    errors.append(f"Failed to save {filename} due to error: {str(e)}")

                finally:
                    file.file.close()
                    
                # load model
                for detector_name in ModelLoader.detectors:
                    detector = ModelLoader.load_detector(model_name=detector_name)
                    det_result = helper.create_aligned_images(
                            file.filename, detector, [])
                    
                    if isinstance(det_result,BaseError):
                            errors.append(det_result)
                            continue;
                    img, faces = det_result

                    for embedder_name in ModelLoader.embedders:
                        embedder = ModelLoader.load_embedder(
                            model_name=embedder_name
                        )
                        # Create cropped images for all faces detected and store them in the respective model folder under static/{model}/

                        # Generate the embeddings for all faces and store them for future indexing
                        result = helper.generate_all_emb(
                            img,
                            faces,
                            file.filename,
                            detector,
                            embedder,
                            gender_age_model,
                        )
                        if isinstance(result,BaseError):
                            errors.append(result)
                            continue;
                        img, face_embeddings,faces=result

                        detector_indices[i][return_detector] = list(
                            range(len(face_embeddings))
                        )
                        manager.add_embedding_typed(
                            face_embeddings, detector_name, embedder_name
                        )

                        if face_embeddings is None:
                            continue

                        generated_embeddings[f"{detector_name}_{embedder_name}"] = (
                            np.array([e.embedding for e in face_embeddings])
                        )

                        # if its the model name that was submitted in the request
                        if (
                            detector_name == return_detector
                            and embedder_name == return_embedder
                        ):
                            # get all the results for the selected model
                            (
                                faces_length.append(len(faces))
                                if faces
                                else faces_length.append(0)
                            )
                            if not faces or len(faces)==0:
                                # if images with no detected faces are allowed save them under the no face directory
                                if save_invalid:
                                    os.replace(
                                        path,
                                        os.path.join(
                                            AppPaths.UPLOAD_FOLDER, "no_face", file.filename
                                        ),
                                    )
                                else:
                                    os.remove(path)
                                invalid_images.append("no_face/" + file.filename)
                                current_images.append(None)
                            else:
                                current_images.append(file.filename)
                                valid_images.append(file.filename)
                        # save the current database state
                    manager.save(detector_name)
                detector_indices[i] = util.get_all_detectors_faces(
                    generated_embeddings, return_detector
                )

            else:
                raise HTTPException(400,f"Invalid file format for {file.filename}.") 

    return UploadImagesResponse(images=valid_images,invalid_images=invalid_images,
                                detector_indices=detector_indices,
                                faces_length=faces_length,errors=errors);


