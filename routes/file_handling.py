from flask import Blueprint, send_from_directory, request, session,jsonify
from modules import ModelLoader,ImageHelper,AppPaths,util
from . import resources
import os
import traceback
import numpy as np
file_handling_bp = Blueprint('file_handling', __name__)

@file_handling_bp.route("/api/upload", methods=[ "POST"])
def upload_image():
    helper=resources.helper
    manager=resources.manager

    errors = []
    current_images = []
    files = request.files.items()
    faces_length = []
    detector_indices: list[dict[str, list[int]]] = []
    valid_images = []
    generated_embeddings: list[dict[str, list[np.ndarray]]] = {}
    # get request parameters
    return_detector = request.form.get("detector_name", "SCRFD10G", type=str)
    return_embedder = request.form.get("embedder_name", "ResNet100GLint360K", type=str)
    gender_age_model = ModelLoader.load_genderage("MobileNetCeleb0.25_CelebA")
    save_invalid = request.form.get("save_invalid", False, type=bool)
    # true if images will be saved without containing faces
    i = -1
    invalid_images = []
    for image_name, file in files:
        detector_indices
        if file and file.filename:
            filename = file.filename.replace("_", "")
            if ImageHelper.allowed_file(file.filename):
                session.pop("uploaded_images", None)
                path = os.path.join(AppPaths.UPLOAD_FOLDER, file.filename)
                try:
                    i += 1
                    detector_indices.append({})
                    # save image
                    file.save(path)
                    # load model
                    for detector_name in ModelLoader.detectors:
                        detector = ModelLoader.load_detector(model_name=detector_name)
                        for embedder_name in ModelLoader.embedders:
                            embedder = ModelLoader.load_embedder(
                                model_name=embedder_name
                            )
                            # Create cropped images for all faces detected and store them in the respective model folder under static/{model}/
                            img, faces, temp_err = helper.create_aligned_images(
                                file.filename, detector, []
                            )
                            # Generate the embeddings for all faces and store them for future indexing
                            img, face_embeddings, temp_err = helper.generate_all_emb(
                                img,
                                faces,
                                file.filename,
                                detector,
                                embedder,
                                gender_age_model,
                            )

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
                            errors = errors + temp_err
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
                                if len(temp_err) > 0 or (not faces):
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
                except Exception as e:
                    tb = traceback.format_exc()
                    errors.append(f"Failed to save {filename} due to error: {str(e)}")

            else:
                errors.append(f"Invalid file format for {file.filename}. ")

    return jsonify(
        {
            "images": valid_images,
            "invalid_images": invalid_images,
            "detector_indices": detector_indices,
            "faces_length": faces_length,
            "errors": errors,
        }
    )


# serve the source images
@file_handling_bp.route("/pool/<path:filename>")
def custom_static(filename):
    return send_from_directory("pool", filename)


# serve the images from the static folder
@file_handling_bp.route("/static_images/<path:filename>")
def processed_static(filename):
    return send_from_directory("static", filename)
