from flask import Blueprint, request, jsonify
from modules import AppPaths,ModelLoader,util,FamilyClassifier
from . import resources 
import numpy as np

image_search_bp = Blueprint('Image Search', __name__)



# compare two images and their selected faces based on their similarity
# if above provided threshold return positive result
@image_search_bp.route("/api/compare", methods=["POST"])
def compare_image():
    helper=resources.helper
    manager=resources.manager

    uploaded_images = request.form.getlist("images")
    similarity_thresh = request.form.get("similarity_thresh", 0.5, type=float)
    detector_name = request.form.get("detector_name", default="SCRFD10G", type=str)
    embedder_name = request.form.get(
        "embedder_name", default="ResNet100GLint360K", type=str
    )
    embedder = ModelLoader.load_embedder(embedder_name)
    detector = ModelLoader.load_detector(detector_name)
    combochanges = [int(x) for x in request.form.getlist("selected_faces")]
    embeddings = []
    messages = []
    errors = []
    # ensure that the user check 2 image faces
    for i in range(len(uploaded_images)):
        if len(uploaded_images) == 2:
            filename = f"aligned_{0 if combochanges[i] == -2 else combochanges[i]}_{uploaded_images[i]}"
            # check if embedding of face already exists
            embedding = manager.get_embedding_by_name(
                filename, detector_name, embedder_name
            ).embedding
            if len(embedding) > 0:
                embeddings.append(embedding)
            else:
                embedding, temp_err = helper.generate_embedding(
                    uploaded_images[i], combochanges[i], detector, embedder
                )
                # Add the errors and embeddings from the helper function to the local variables
                errors = errors + temp_err
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    print("No embedding extracted.")  # Debug log
        else:
            errors.append("Select exactly 2 images for comparison.")

    if len(embeddings) == 2:
        # Calculate the  similarity between the two selected faces images and return the result based in the configured threshold
        similarity = util.calculate_similarity(embeddings[0], embeddings[1])
        messages.append(f"Similarity: {similarity:.4f}")
        # if similarity >= similarity_thresh:
        # messages.append(f"THIS IS PROBABLY THE SAME PERSON ")
        # else:
        # messages.append(f"THIS IS PROBABLY NOT THE SAME PERSON")

    elif len(uploaded_images) != 2:
        errors.append("choose 2 images!")
    else:
        errors.append("Error: Failed to extract embeddings from images.")

    return jsonify(
        {
            "errors": errors,
            "messages": messages,
        }
    )

@image_search_bp.route("/api/check_family", methods=["POST"])
def checkisfamily():
    helper=resources.helper
    manager=resources.manager
    detector_name = request.form.get("detector_name", default="SCRFD10G", type=str)
    embedder_name = request.form.get(
        "embedder_name", default="ResNet100GLint360K", type=str
    )
    # similarity_thresh=request.form.get("similarity_thresh",0.6,type=float)
    gender_age = ModelLoader.load_genderage("MobileNetCeleb0.25_CelebA")
    embedder = ModelLoader.load_embedder(embedder_name)
    detector = ModelLoader.load_detector(detector_name)
    uploaded_images = request.form.getlist("images")
    # helper.cluster_family_images(model_name,APP_DIR,uploaded_images,model)
    combochanges = [int(x) for x in request.form.getlist("selected_faces")]
    embeddings = []
    messages = []
    errors = []
    # check if the user upload 2 face images
    if len(uploaded_images) == 2:
        for i in range(len(uploaded_images)):
            # check if first name embedding already exists in repository
            aligned_filename = f"aligned_{0 if combochanges[i] == -2 else combochanges[i]}_{uploaded_images[i]}"

            embedding = manager.get_embedding_by_name(
                aligned_filename, detector_name, embedder_name
            ).embedding
            # embedding=ImageHelper.extract_embedding(uploaded_images[0])
            if len(embedding) > 0:
                embeddings.append(embedding)
            else:
                embedding, temp_err = helper.generate_embedding(
                    uploaded_images[i], 0, detector, embedder
                )
                # Add the errors and embeddings from the helper function to the local variables
                errors = errors + temp_err
                if embedding is not None:
                    embeddings.append(embedding)
                else:
                    print("No embedding extracted.")  # Debug log
    else:
        errors.append("Select exactly 2 images for comparison.")

    if len(embeddings) == 2:
        # Calculate similarity between the two images
        similarity = util.calculate_similarity(embeddings[0], embeddings[1])
        classifier = FamilyClassifier(AppPaths.APP_DIR)
        Genders = [0, 0]

        for i in range(2):
            img, faces = helper._ImageHelper__extract_faces(
                uploaded_images[i], detector
            )
            if faces:
                Genders[i],_ = gender_age.get_gender_age(img, faces[0])

        is_same_family, similar = classifier.predict(similarity, Genders[0], Genders[1])
        messages.append(
            f"Probably the Same Family with similarity {similar:.2f}"
            if is_same_family
            else f"Probably Not the Same Family with similarity {similar:.2f}"
        )

    return jsonify(
        {
            "errors": errors,
            "messages": messages,
        }
    )


@image_search_bp.route("/api/check_many", methods=["POST"])
def find_similar_images():
    helper=resources.helper
    errors = []
    messages = []
    detector_name = request.form.get("detector_name", default="SCRFD10G", type=str)
    embedder_name = request.form.get(
        "embedder_name", default="ResNet100GLint360K", type=str
    )
    embedder = ModelLoader.load_embedder(embedder_name)
    detector = ModelLoader.load_detector(detector_name)
    similarity_thresh = request.form.get("similarity_thresh", 0.5, type=float)
    current_image = request.form.get("image")
    selected_face = int(request.form.get("selected_face"))
    k = int(request.form.get("number_of_images", 5))
    similar, temp_err = helper.get_k_similar_images(
        current_image, selected_face, similarity_thresh, detector, embedder, k
    )
    errors = errors + temp_err
    json_similar = [s.to_json() for s in similar]
    return jsonify({"images": json_similar, "errors": errors})


@image_search_bp.route("/api/check", methods=["POST"])
def find_similar_image():
    helper=resources.helper
    manager=resources.manager
    base_detector_name = request.form.get("detector_name", default="SCRFD10G", type=str)
    base_embedder_name = request.form.get(
        "embedder_name", default="ResNet100GLint360K", type=str
    )
    similarity_thresh = request.form.get("similarity_thresh", 0.5, type=float)
    base_embedder = ModelLoader.load_embedder(base_embedder_name)
    base_detector = ModelLoader.load_detector(base_detector_name)

    most_similar_image = None
    messages = []
    errors = []
    image_name = ""
    face_num = -2
    face_length = 0
    current_image = request.form.get("image")
    selected_face = int(request.form.get("selected_face"))
    if len(current_image) > 0:
        similar_images, temp_err = helper.get_k_similar_images(
            current_image,
            selected_face,
            similarity_thresh,
            detector=base_detector,
            embedder=base_embedder,
            k=5,
        )
        errors = errors + temp_err
        if len(similar_images) > 0:
            most_similar_image = max(similar_images, key=lambda x: x.similarity)
    else:
        errors.append("no images selected for check")
    if most_similar_image:

        messages.append(
            f"The most similar face is no. {most_similar_image.face_num+1} in image {most_similar_image.image_name} with similarity of {most_similar_image.similarity:.4f}"
        )
        face_length = len(
            helper.get_face_boxes(
                most_similar_image.image_name, base_detector_name, base_embedder_name
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
                img, faces, temp_err = helper.create_aligned_images(
                    image_name, temp_detector, []
                )

                _, new_embs, _ = helper.generate_all_emb(
                    img, faces, image_name, temp_detector, temp_embedder
                )
                manager.add_embedding_typed(
                        new_embs, detector_name, embedder_name
                    )
                embs=np.array([f.embedding for f in new_embs])
            generated_embeddings[f"{detector_name}_{embedder_name}"] = embs

        detector_indices = util.get_all_detectors_faces(
            generated_embeddings, base_detector_name
        )
        return jsonify(
            {
                "image": image_name,
                "face": face_num,
                "face_length": face_length,
                "detector_indices": detector_indices,
                "errors": errors,
                "messages": messages,
            }
        )
    return jsonify(
        {
            "image": "",
            "face": -2,
            "face_length": 0,
            "detector_indices": [],
            "errors": errors,
            "messages": messages,
        }
    )


@image_search_bp.route("/api/check_template", methods=["POST"])
def find_by_template():
    helper=resources.helper
   
    most_similar_image = None
    messages = []
    errors = []
    box = []
    similarity = -1
    current_image = request.form.get("template")
    similarity_thresh = request.form.get("similarity_thresh", 0.5, type=float)
    if len(current_image) > 0:
        (
            most_similar_image,
            box,
            similarity,
            temp_err,
        ) = helper.get_most_similar_image_by_template(current_image, similarity_thresh)
        errors = errors + temp_err
    else:
        errors.append("no images selected for check")
    if most_similar_image:
        messages.append(
            f"The most similar image is {most_similar_image} with similarity of {similarity:.4f}"
        )
    return jsonify(
        {
            "image": most_similar_image,
            "box": box,
            "errors": errors,
            "messages": messages,
        }
    )


@image_search_bp.route("/api/filter", methods=["POST"])
def filter():
    helper=resources.helper
    detector_name = request.form.get("detector_name", default="SCRFD10G", type=str)
    embedder_name = request.form.get(
        "embedder_name", default="ResNet100GLint360K", type=str
    )
    threshold = float(request.form.get("threshold", 999))
    if threshold < 1:
        deleted = helper.filter(threshold,detector_name,embedder_name)
        return jsonify({"success": True, "deleted": deleted})
    return jsonify({"success": False, "error": "threshold must be below 1"})
