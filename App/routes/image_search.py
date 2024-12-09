from Shared.services import util
from Shared.services.models.model_loader import ModelLoader
from App.models.requests import CompareFacesRequest,CompareKinshipRequest,SearchSimilarRequest,SearchMostSimilarRequest
from App.models.responses import CompareFacesResponse,CompareKinshipResponse,SearchSimilarResponse,SearchMostSimilarResponse,NoMatchResponse
from . import resources 
from Shared.models.embedder_name import EmbedderName
import numpy as np
import os
from Shared.models.stored_embedding import FaceEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from routes.resources import Container
from Shared.services.stores.embeddings import BaseImageEmbeddingManager
from Shared.services.stores.metadata_manager import MetadataManager
from dependency_injector.wiring import inject, Provide
from fastapi import APIRouter,HTTPException,Depends
from Shared.models.errors.base_error import BaseError
from fastapi.encoders import jsonable_encoder
from Shared.services.processing.face_aligner import FaceAligner
from Shared.services.processing.face_similarity_search import FaceSimilaritySearch
from Shared.services.processing.local.local_embedding_generator import LocalEmbeddingGenerator
from Shared.services.stores.media.base_image_storage import BaseImageStorage
import tempfile
image_search_router=APIRouter()

# compare two images and their selected faces based on their similarity
# if above provided threshold return positive result
@image_search_router.post("/api/similarity/compare")
@inject
async def compare_image(request:CompareFacesRequest,
                        face_aligner:FaceAligner=Depends(Provide[Container.face_aligner]),
                        embedding_generator:LocalEmbeddingGenerator=Depends(Provide[Container.embedding_generator]),
                        emb_manager:BaseImageEmbeddingManager=Depends(Provide[Container.emb_manager]),
                        image_storage:BaseImageStorage=Depends(Provide[Container.image_storage]),
                        model_loader:ModelLoader=Depends(Provide[Container.default_model_loader])
                        )->CompareFacesResponse:

    
    uploaded_images = request.images

    embedder = model_loader.load_embedder(request.embedder_name)
    detector = model_loader.load_detector(request.detector_name)
    combochanges = request.selected_faces
    similarity=-1
    embeddings = []

    # ensure that the user check 2 image faces
    for i in range(len(uploaded_images)):
        if len(uploaded_images) == 2 and len(combochanges)==2:
            filename= util.face_path(uploaded_images[i], combochanges[i])
            # check if embedding of face already exists
            embedding = emb_manager.get_embedding_by_name(
                filename, request.detector_name, request.embedder_name
            )
            if embedding is not None and len(embedding.embedding)>0:
                embeddings.append(embedding.embedding)
            else:
                
                img,file_path=image_storage.load_image(uploaded_images[i],detector.name)
                faces_dir=tempfile.TemporaryDirectory(prefix=uploaded_images[i],suffix="_faces")

                det_result = face_aligner.create_aligned_images(
                    file_path,save_file_name=uploaded_images[i] ,img=img,faces_dir=faces_dir.name,detector=detector
                )
                if(isinstance(det_result,BaseError)):
                    os.rmdir(faces_dir.name)
                    raise HTTPException(500,jsonable_encoder(det_result)) 
                
                img, faces =det_result
                result = embedding_generator.generate_all_emb(
                    img, faces, uploaded_images[i], detector, embedder
                )
                if(isinstance(result,BaseError)):
                    os.rmdir(faces_dir.name)
                    raise HTTPException(500,jsonable_encoder(result)) 

                img, new_embs ,faces=result
                
                emb_manager.add_embedding_typed(
                    new_embs, request.detector_name, request.embedder_name
                )
                emb=[f.embedding for f in new_embs if f.name==filename]

                embeddings.append(emb[0])

                # Add the errors and embeddings from the helper function to the local variables
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    print("No embedding extracted.")  # Debug log
        else:
            raise HTTPException(400,"Choose 2 Faces to compare!")

    if len(embeddings) == 2:
        # Calculate the  similarity between the two selected faces images and return the result based in the configured threshold
        similarity = util.calculate_similarity(embeddings[0], embeddings[1])
    else:
        raise HTTPException(500,"Failed to extract embeddings from images.")
    if similarity>-1:
        response=CompareFacesResponse(similarity=similarity);
        return response


@image_search_router.post("/api/similarity/compare_kinship")
@inject
async def compare_kinship(request:CompareKinshipRequest,
                        face_similarity_search:FaceSimilaritySearch=Depends(Provide[Container.face_similarity_search]),
                        face_aligner:FaceAligner=Depends(Provide[Container.face_aligner]),
                        embedding_generator:LocalEmbeddingGenerator=Depends(Provide[Container.embedding_generator]),
                        emb_manager:BaseImageEmbeddingManager=Depends(Provide[Container.emb_manager]),
                        image_storage:BaseImageStorage=Depends(Provide[Container.image_storage]),
                        model_loader:ModelLoader=Depends(Provide[Container.default_model_loader])
                        )->CompareKinshipResponse:

    detector_name = request.detector_name
    embedder_name = request.embedder_name
    kinship_embedder_names = request.kinship_embedder_names
    similarity_thresh = request.similarity_threshold
    quality_thresh = request.quality_threshold

    embedder = model_loader.load_embedder(embedder_name)
    detector = model_loader.load_detector(detector_name)
    uploaded_images = request.images
    # helper.cluster_family_images(model_name,APP_DIR,uploaded_images,model)
    image_count= len(uploaded_images)
    combochanges = request.selected_faces
    kinship_embeddings:dict[EmbedderName, list[list[FaceEmbedding]]]={kinship_embedder_name:[[] for _ in range(image_count)] 
                                                                      for kinship_embedder_name in kinship_embedder_names}
    kinship_similarities={}
    # check if the user uploaded 2 face images
    if image_count == 2:
        for i in range(image_count):
            # check if first name embedding already exists in repository
            face_num=0 if combochanges[i] == -2 else combochanges[i]
            aligned_filename = util.face_path(uploaded_images[i],face_num)

            existing_embedding = emb_manager.get_embedding_by_name(
                aligned_filename, detector_name, embedder_name
            )
            if(existing_embedding is None):
                img,file_path=image_storage.load_image(uploaded_images[i],detector.name)
                faces_dir=tempfile.TemporaryDirectory(prefix=uploaded_images[i],suffix="_faces")

                det_result = face_aligner.create_aligned_images(
                    file_path,save_file_name=uploaded_images[i] ,img=img,faces_dir=faces_dir.name,detector=detector
                )
                if(isinstance(det_result,BaseError)):
                    os.rmdir(faces_dir.name)
                    raise HTTPException(500,jsonable_encoder(det_result)) 
                
                img, faces =det_result
                result = embedding_generator.generate_all_emb(
                    img, faces, uploaded_images[i], detector, embedder
                )
                if(isinstance(result,BaseError)):
                    os.rmdir(faces_dir.name)
                    raise HTTPException(500,jsonable_encoder(result)) 

                img, new_embs ,faces=result
                emb_manager.add_embedding_typed(
                        new_embs, detector_name, embedder_name
                    )
                embedding=new_embs[face_num]
            else:
                embedding=existing_embedding

            similar_images=face_similarity_search.get_similar_images(embedding.embedding,aligned_filename,detector_name,embedder_name,100,quality_thresh);
            for kinship_embedder_name in kinship_embedder_names:
                
                for image in similar_images:
                    if image["similarity"]>=similarity_thresh:
                        #same person
                        embedding_get_result=emb_manager.get_embedding_by_name(image["Embedding"].name,detector_name,kinship_embedder_name)
                        if(embedding_get_result is not None):
                            kinship_embedding=embedding_get_result.embedding
                            kinship_embeddings[kinship_embedder_name][i].append(kinship_embedding)
                    else:
                        #it comes sorted by highest similarity first, if its not similar enough the rest are not either
                        break;
                kinship_embeddings[kinship_embedder_name][i].append(emb_manager.get_embedding_by_name(embedding.name,detector_name,kinship_embedder_name).embedding);

        for kinship_embedder_name in kinship_embedder_names:
            kinship_similarity_matrix=cosine_similarity(kinship_embeddings[kinship_embedder_name][0],kinship_embeddings[kinship_embedder_name][1])
            kinship_similarities[kinship_embedder_name] = np.mean(kinship_similarity_matrix)
            print(kinship_similarity_matrix)

    response =CompareKinshipResponse(average_similarities=kinship_similarities,
                                     cluster1_count=len(kinship_embeddings[kinship_embedder_name][0]),
                                     cluster2_count=len(kinship_embeddings[kinship_embedder_name][1]))
    return response;

@image_search_router.post("/api/similarity/search_many")
@inject
async def find_similar_images(request:SearchSimilarRequest,
                        face_similarity_search:FaceSimilaritySearch=Depends(Provide[Container.face_similarity_search]),
                        model_loader:ModelLoader=Depends(Provide[Container.default_model_loader])
                        )->SearchSimilarResponse:
    detector_name = request.detector_name
    embedder_name = request.embedder_name
    embedder = model_loader.load_embedder(embedder_name)
    detector = model_loader.load_detector(detector_name)
    similarity_thresh = request.similarity_threshold
    quality_thresh = request.quality_threshold
    current_image = request.image
    selected_face = request.selected_face
    k = request.number_of_images

    similar = face_similarity_search.get_k_similar_images(
        current_image, selected_face, similarity_thresh, detector, embedder, k,quality_thresh
    )
    if isinstance(similar,BaseError):
        return SearchSimilarResponse(images=[],errors=similar) 
    response=SearchSimilarResponse(images=similar)
    return response


@image_search_router.post("/api/similarity/search")
@inject
async def find_similar_image(request:SearchMostSimilarRequest,
                              face_similarity_search:FaceSimilaritySearch=Depends(Provide[Container.face_similarity_search]),
                        metadata_manager:MetadataManager=Depends(Provide[Container.metadata_manager]),
                        emb_manager:BaseImageEmbeddingManager=Depends(Provide[Container.emb_manager]),
                         model_loader:ModelLoader=Depends(Provide[Container.default_model_loader]) )->SearchMostSimilarResponse|NoMatchResponse:

    base_detector_name = request.detector_name
    base_embedder_name = request.embedder_name
    base_embedder = model_loader.load_embedder(base_embedder_name)
    base_detector = model_loader.load_detector(base_detector_name)
    similarity_thresh = request.similarity_threshold
    quality_thresh = request.quality_threshold
    current_image = request.image
    selected_face = request.selected_face

    most_similar_image = None
    image_name = ""
    face_num = -2
    face_length = 0
    if len(current_image)==0:
        raise HTTPException(400,detail="no images selected for check")
    if len(current_image) > 0:
        similar_images = face_similarity_search.get_k_similar_images(
            current_image,
            selected_face,
            similarity_thresh,
            detector=base_detector,
            embedder=base_embedder,
            k=5,
            quality_thresh=quality_thresh
        )
        if isinstance(similar_images,BaseError):
            raise HTTPException(500,detail=jsonable_encoder(similar_images))
        if len(similar_images) > 0:
            most_similar_image = max(similar_images, key=lambda x: x.similarity)

    if most_similar_image:

        face_length = len(
            metadata_manager.get_image_faces(
                most_similar_image.image_name,-2, base_detector_name, base_embedder_name
            )
        )
        image_name = most_similar_image.image_name
        face_num = most_similar_image.face_num
        generated_embeddings = {}
        embedder_name = next(iter(model_loader.model_registry["embedders"]))
        for detector_name in model_loader.model_registry["detectors"]:
            embs = emb_manager.get_image_embeddings(
                image_name, detector_name, embedder_name
            )
            if len(embs) == 0:
                temp_detector=model_loader.load_detector(detector_name)
                temp_embedder=model_loader.load_embedder(embedder_name)
                
                det_result = face_similarity_search.face_aligner.create_aligned_images(
                    image_name, temp_detector
                )
                if isinstance(det_result,BaseError):
                    raise HTTPException(500,detail=det_result)
                img, faces = det_result
                result = face_similarity_search.embedding_generator.generate_all_emb(
                    img, faces, image_name, temp_detector, temp_embedder
                )
                if isinstance(result,BaseError):
                    raise HTTPException(500,detail=result)
                _, new_embs,_=result

                emb_manager.add_embedding_typed(
                        new_embs, detector_name, embedder_name
                    )
                embs=np.array([f.embedding for f in new_embs])
            generated_embeddings[f"{detector_name}_{embedder_name}"] = embs

        indices_result = util.get_all_detectors_faces(
            image_name,generated_embeddings, base_detector_name,model_loader,metadata_manager
        )
        if isinstance(indices_result,BaseError):
            return indices_result
        detector_indices,detector_metadata =indices_result
        response=SearchMostSimilarResponse(image=image_name,face=face_num,face_length=face_length,
                                           detector_indices=detector_indices,
                                           metadata=detector_metadata,
                                           similarity=most_similar_image.similarity)
        return response;

    return NoMatchResponse()

@image_search_router.post("/api/similarity/filter")
@inject
def filter(request:SearchMostSimilarRequest):
    helper=resources.helper
    detector_name = request.detector_name
    embedder_name = request.embedder_name

    threshold = request.similarity_threshold
    if threshold < 1:
        deleted = helper.filter(threshold,detector_name,embedder_name)
        return {"success": True, "deleted": deleted}
    return {"success": False, "error": "threshold must be below 1"}
