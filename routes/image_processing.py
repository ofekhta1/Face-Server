from services import ModelLoader
from config.app_paths import AppPaths
from . import resources
import os
import cv2
from fastapi import APIRouter,HTTPException
from models.requests import ProcessImagesRequest

image_processing_router=APIRouter()

@image_processing_router.post("/api/improve")
def improve_image(request):
    helper=resources.helper

    image = request.form.get("image")
    detector_name = request.form.get("detector_name", default="SCRFD10G", type=str)
    embedder_name = request.form.get(
        "embedder_name", default="ResNet100GLint360K", type=str
    )
    detector = ModelLoader.load_detector(detector_name)
    embedder = ModelLoader.load_embedder(embedder_name)

    enhanced_image = image
    errors = []
    messages = []
    i = 0
    print(request.form)
    # check if the image is already enhanced
    if "enhanced" not in image:
        try:
            enhanced_img = helper.enhance_image(image)
            if enhanced_img is not None:
                enhanced_image_path = os.path.join(AppPaths.UPLOAD_FOLDER, "enhanced_" + image)
                cv2.imwrite(enhanced_image_path, enhanced_img)
                enhanced_image = "enhanced_" + image

                img, faces, temp_err = helper.create_aligned_images(
                    "enhanced_" + image, detector, []
                )
                img, faces, embeddings, aligned_images, temp_err = (
                    helper.generate_all_emb(
                        img, faces, "enhanced_" + image, detector, embedder
                    )
                )
                helper.save_embeddings(
                    faces, embeddings, aligned_images, detector_name, embedder_name
                )
                errors = errors + temp_err

                if len(temp_err) > 0:
                    os.remove(enhanced_image_path)
        except Exception as e:
            errors.append(str(e))
    else:
        errors.append(f"image {image} is already enhanced!")

    return {
        "enhanced_image": enhanced_image,
        "errors": errors,
        "messages": messages,
    }


@image_processing_router.post("/api/align")
def align_image(request:ProcessImagesRequest):
    helper=resources.helper
    
    uploaded_images = request.images
    detector_name = request.detector_name
    detector = ModelLoader.load_detector(detector_name)
    faces_length = []
    errors = []
    images=[]
    # align the images if they aren't already aligned
    for i in range(len(uploaded_images)):
        filename = uploaded_images[i]
        if "aligned" in filename or "detected" in filename:
            path = os.path.join(AppPaths.STATIC_FOLDER, filename)
        else:
            path = os.path.join(AppPaths.UPLOAD_FOLDER, filename)
        if os.path.exists(path):
            _, faces = helper.create_aligned_images(filename, detector, images)
            faces_length.append(len(faces))
        else:
            errors.append(f"File {filename} does not exist!")
    return {
        
            "images": images,
            "faces_length": faces_length,
            "errors": errors,
        }


# creates detected images for faces for all uploaded images for a specific model
# the detected images contain kps (5 points)
@image_processing_router.post("/api/detect")
def detect_image(request:ProcessImagesRequest):
    helper=resources.helper
    uploaded_images = request.images
    detector_name = request.detector_name
    detector = ModelLoader.load_detector(detector_name)
    faces_length = []
    errors = []
    images = []
    for i in range(len(uploaded_images)):
        filename = uploaded_images[i]
        if "aligned" in filename or "detected" in filename:
            path = os.path.join(AppPaths.STATIC_FOLDER, detector_name, filename)
        else:
            path = os.path.join(AppPaths.UPLOAD_FOLDER, filename)
        if os.path.exists(path):
            # ensure that the user check img with faces for the detection
            face_count, _ = helper.detect_faces_in_image(filename, detector, images)

            if face_count is not None:
                faces_length.append(face_count)
            else:
                errors.append(
                    f"you can't detect this File {filename} beacuse it does not contain a face!"
                )
        else:
            errors.append(f"File {filename} does not exist!")
    return {
            "images": images,
            "faces_length": faces_length,
            "errors": errors,
        }
