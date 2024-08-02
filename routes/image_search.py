from modules import AppPaths,ModelLoader,util,FamilyClassifier
from models.requests import CompareFacesRequest,CompareKinshipRequest,SearchSimilarRequest,SearchMostSimilarRequest
from models.responses import CompareFacesResponse,CompareKinshipResponse,SearchSimilarResponse,SearchMostSimilarResponse,NoMatchResponse
from . import resources 
from models.embedder_name import EmbedderName
import numpy as np
from models.stored_embedding import FaceEmbedding
from sklearn.metrics.pairwise import cosine_similarity
from fastapi import APIRouter,HTTPException
from models.errors.base_error import BaseError
from fastapi.encoders import jsonable_encoder


image_search_router=APIRouter()

# compare two images and their selected faces based on their similarity
# if above provided threshold return positive result
@image_search_router.post("/api/compare")
async def compare_image(request:CompareFacesRequest)->CompareFacesResponse:
    helper=resources.helper
    manager=resources.manager
    
    
    uploaded_images = request.images

    embedder = ModelLoader.load_embedder(request.embedder_name)
    detector = ModelLoader.load_detector(request.detector_name)
    combochanges = request.selected_faces
    similarity=-1
    embeddings = []

    # ensure that the user check 2 image faces
    for i in range(len(uploaded_images)):
        if len(uploaded_images) == 2 and len(combochanges)==2:
            filename = f"aligned_{0 if combochanges[i] == -2 else combochanges[i]}_{uploaded_images[i]}"
            # check if embedding of face already exists
            embedding = manager.get_embedding_by_name(
                filename, request.detector_name, request.embedder_name
            )
            if embedding is not None and len(embedding.embedding)>0:
                embeddings.append(embedding.embedding)
            else:
                det_result = helper.create_aligned_images(
                        uploaded_images[i], detector, [])
                
                if isinstance(det_result,BaseError):
                        raise HTTPException(500,det_result) 

                img, faces = det_result
                
                result= helper.generate_all_emb(
                    img, faces, uploaded_images[i], detector, embedder
                )


                if isinstance(result,BaseError):
                    raise HTTPException(500,result) 

                img, new_embs ,faces=result
                
                manager.add_embedding_typed(
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


@image_search_router.post("/api/compare_kinship")
def compare_kinship(request:CompareKinshipRequest)->CompareKinshipResponse:
    helper=resources.helper
    manager=resources.manager
    detector_name = request.detector_name
    embedder_name = request.embedder_name
    kinship_embedder_name = request.kinship_embedder_name
    similarity_thresh = request.similarity_threshold
    quality_thresh = request.quality_threshold
    if(not EmbedderName.is_kinship(kinship_embedder_name)):
        raise HTTPException(400,jsonable_encoder(BaseError(reason=f"{kinship_embedder_name} is not a valid kinship embedder name")))

    embedder = ModelLoader.load_embedder(embedder_name)
    detector = ModelLoader.load_detector(detector_name)
    uploaded_images = request.images
    # helper.cluster_family_images(model_name,APP_DIR,uploaded_images,model)
    image_count= len(uploaded_images)
    combochanges = request.selected_faces
    embeddings:list[list[FaceEmbedding]] = [[] for _ in range(image_count)]

    # check if the user uploaded 2 face images
    if image_count == 2:
        for i in range(image_count):
            # check if first name embedding already exists in repository
            face_num=0 if combochanges[i] == -2 else combochanges[i]
            aligned_filename = f"aligned_{face_num}_{uploaded_images[i]}"

            existing_embedding = manager.get_embedding_by_name(
                aligned_filename, detector_name, embedder_name
            )
            if(existing_embedding is None):
                det_result = helper.create_aligned_images(
                    uploaded_images[i], detector, []
                )
                if(isinstance(det_result,BaseError)):
                    raise HTTPException(500,jsonable_encoder(det_result));
                
                img, faces = det_result
                result = helper.generate_all_emb(
                    img, faces, uploaded_images[i], detector, embedder
                )
                if(isinstance(result,BaseError)):
                    raise HTTPException(500,jsonable_encoder(result));

                _, new_embs =result
                manager.add_embedding_typed(
                        new_embs, detector_name, embedder_name
                    )
                embedding=new_embs[face_num]
            else:
                embedding=existing_embedding

            similar_images=helper.get_similar_images(embedding.embedding,aligned_filename,detector_name,embedder_name,100,quality_thresh);
            for image in similar_images:
                if image["similarity"]>=similarity_thresh:
                    #same person
                    embedding_get_result=helper.emb_manager.get_embedding_by_name(image["Embedding"].name,detector_name,kinship_embedder_name)
                    if(embedding_get_result is not None):
                        kinship_embedding=embedding_get_result.embedding
                        embeddings[i].append(kinship_embedding)
            embeddings[i].append(helper.emb_manager.get_embedding_by_name(embedding.name,detector_name,kinship_embedder_name).embedding);

        kinship_similarity_matrix=cosine_similarity(embeddings[0],embeddings[1])
        average_similarity = np.mean(kinship_similarity_matrix)
        print(kinship_similarity_matrix)

    response =CompareKinshipResponse(average_similarity=average_similarity,cluster1_count=len(embeddings[0]),cluster2_count=len(embeddings[1]))
    return response;

@image_search_router.post("/api/check_many")
def find_similar_images(request:SearchSimilarRequest)->SearchSimilarResponse:
    helper=resources.helper
    detector_name = request.detector_name
    embedder_name = request.embedder_name
    embedder = ModelLoader.load_embedder(embedder_name)
    detector = ModelLoader.load_detector(detector_name)
    similarity_thresh = request.similarity_threshold
    quality_thresh = request.quality_threshold
    current_image = request.image
    selected_face = request.selected_face
    k = request.number_of_images

    similar = helper.get_k_similar_images(
        current_image, selected_face, similarity_thresh, detector, embedder, k,quality_thresh
    )
    if isinstance(similar,BaseError):
        return SearchSimilarResponse(images=[],errors=similar) 
    response=SearchSimilarResponse(images=similar)
    return response


@image_search_router.post("/api/check")
def find_similar_image(request:SearchMostSimilarRequest)->SearchMostSimilarResponse|NoMatchResponse:
    helper=resources.helper
    manager=resources.manager
    base_detector_name = request.detector_name
    base_embedder_name = request.embedder_name
    base_embedder = ModelLoader.load_embedder(base_embedder_name)
    base_detector = ModelLoader.load_detector(base_detector_name)
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
        similar_images = helper.get_k_similar_images(
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
            helper.get_image_faces(
                most_similar_image.image_name,-2, base_detector_name, base_embedder_name
            )
        )
        image_name = most_similar_image.image_name
        face_num = most_similar_image.face_num
        generated_embeddings = {}
        embedder_name = next(iter(ModelLoader.embedders))
        for detector_name in ModelLoader.detectors:
            embs = helper.emb_manager.get_image_embeddings(
                image_name, detector_name, embedder_name
            )
            if len(embs) == 0:
                temp_detector=ModelLoader.load_detector(detector_name)
                temp_embedder=ModelLoader.load_embedder(embedder_name)
                
                det_result = helper.create_aligned_images(
                    image_name, temp_detector, []
                )
                if isinstance(det_result,BaseError):
                    raise HTTPException(500,detail=det_result)
                img, faces = det_result
                result = helper.generate_all_emb(
                    img, faces, image_name, temp_detector, temp_embedder
                )
                if isinstance(result,BaseError):
                    raise HTTPException(500,detail=result)
                _, new_embs=result

                manager.add_embedding_typed(
                        new_embs, detector_name, embedder_name
                    )
                embs=np.array([f.embedding for f in new_embs])
            generated_embeddings[f"{detector_name}_{embedder_name}"] = embs

        detector_indices = util.get_all_detectors_faces(
            generated_embeddings, base_detector_name
        )

        response=SearchMostSimilarResponse(image=image_name,face=face_num,face_length=face_length,
                                           detector_indices=detector_indices,
                                           similarity=most_similar_image.similarity)
        return response;

    return NoMatchResponse()


# @image_search_bp.route("/api/check_template", methods=["POST"])
# def find_by_template():
#     helper=resources.helper
   
#     most_similar_image = None
#     messages = []
#     errors = []
#     box = []
#     similarity = -1
#     current_image = request.form.get("template")
#     similarity_thresh = request.form.get("similarity_thresh", 0.5, type=float)
#     if len(current_image) > 0:
#         (
#             most_similar_image,
#             box,
#             similarity,
#             temp_err,
#         ) = helper.get_most_similar_image_by_template(current_image, similarity_thresh)
#         errors = errors + temp_err
#     else:
#         errors.append("no images selected for check")
#     if most_similar_image:
#         messages.append(
#             f"The most similar image is {most_similar_image} with similarity of {similarity:.4f}"
#         )
#     return jsonify(
#         {
#             "image": most_similar_image,
#             "box": box,
#             "errors": errors,
#             "messages": messages,
#         }
#     )


@image_search_router.post("/api/filter")
def filter(request:SearchMostSimilarRequest):
    helper=resources.helper
    detector_name = request.detector_name
    embedder_name = request.embedder_name

    threshold = request.similarity_threshold
    if threshold < 1:
        deleted = helper.filter(threshold,detector_name,embedder_name)
        return {"success": True, "deleted": deleted}
    return {"success": False, "error": "threshold must be below 1"}
