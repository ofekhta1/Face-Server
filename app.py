from flask import Flask, render_template, request, session, url_for, redirect, jsonify
import os
import cv2
import requests as req
import numpy as np
from image_helper import ImageHelper
from image_embedding_manager import ImageEmbeddingManager
from model_loader import ModelLoader
from flask_cors import CORS, cross_origin
from insightface.utils.face_align import norm_crop
from flask import send_from_directory
APP_DIR = os.path.dirname(__file__)
from PIL import Image
UPLOAD_FOLDER = r'C:\python\images_from_site'
#UPLOAD_FOLDER = os.path.join(APP_DIR, "pool")
STATIC_FOLDER = os.path.join(APP_DIR, "static")
# Create the folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)
check_uploaded_images=[]
images = []
def alignforcheck(selected_face,filename,images):
    path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    img = cv2.imread(path)
    faces = detector.get(img)
    for face in faces:
             landmarks = face['kps'].astype(np.int) 
             aligned_filename = f"aligned_{selected_face}_{filename}"   # Name contains the selected face index
             aligned_path =os.path.join(STATIC_FOLDER, aligned_filename)
             aligned_img = norm_crop(img, landmarks,112,'arcface')                            
             cv2.imwrite(aligned_path, aligned_img)
             images.append(aligned_filename)
             uploaded_images = images
             return uploaded_images
def calculate_similarity(emb_a, emb_b):
    similarity = np.dot(emb_a, emb_b) / (np.linalg.norm(emb_a) * np.linalg.norm(emb_b))
    return similarity


def extract_embedding(embedder, face_data):
    try:        
        if face_data and 'embedding' in face_data:
            embedding = face_data['embedding']
            return embedding
        else:
            print("No faces detected.")  # Debug log
            return None
    except Exception as e:
        print("Error during embedding extraction:", e)  # Debug log
        return None

app = Flask(__name__, static_folder=STATIC_FOLDER)

cors = CORS(app)
app.config['CORS_HEADERS'] = 'Content-Type'
@app.route("/pool/<path:filename>")
def custom_static(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)


@app.route("/static_images/<path:filename>")
def processed_static(filename):
    return send_from_directory(STATIC_FOLDER, filename)
def check_uploaded_images(filename, image_helper_instance):
    # Check if file extension is allowed
    if not image_helper_instance.allowed_file(filename):
        return False, "File type not allowed."
    # Check image size
    image_path = os.path.join(image_helper_instance.UPLOAD_FOLDER, filename)
    image = cv2.imread(image_path)
    if image is None:
        return False, "Invalid image file."

    height, width, _ = image.shape
    if height > 2000 or width > 2000:  # Set dimensions as per your requirements
        return False, "Image dimensions are too large."

    # Check if the image contains a face
    detected_faces_count = image_helper_instance.detect_faces_in_image(filename, [])

    if detected_faces_count == 0:
        return False, "No faces detected in the uploaded image."

    # If all checks pass, add the image filename to the session
    if 'uploaded_images' not in session:
        session['uploaded_images'] = []

    session['uploaded_images'].append(filename)
    session.modified = True

    return True, "Image uploaded and validated successfully."


@app.route("/api/upload", methods=["POST"])
def upload_image():
    images=[]
    errors = []
    faces_length = [0, 0]
    current_images = []
    for image_name in ["image1", "image2"]:
        file = request.files.get(image_name)

        if file and file.filename:
            if ImageHelper.allowed_file(file.filename):
                session.pop("uploaded_images", None)
                path = os.path.join(UPLOAD_FOLDER, file.filename)
                try:
                    file.save(path)
                    # Generate the embeddings for all faces and store them for future indexing
                    temp_err=helper.generate_all_emb(path,file.filename);
                    errors=errors+temp_err;

                    if(len(errors)>0):
                        os.remove(path);
                    else:
                        current_images.append(file.filename)
                        images.append(file.filename)
                        # if file and ImageHelper.allowed_file(file.filename):
                        #  filename = file.filename
                        # file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
                        # check_uploaded_images(filename, helper)
                        # session['check_uploaded_images'] = check_uploaded_images
                except Exception as e:
                    errors.append(
                        f"Failed to save {file.filename} due to error: {str(e)}"
                    )
                
            else:
                errors.append(f"Invalid file format for {file.filename}. ")

    if(len(errors)==0):
    # Detect faces immediately after uploading to fill the combo box
        for i in range(len(current_images)):
            faces_length[i] = helper.create_aligned_images(current_images[i], images)
    return jsonify({"images": images, "faces_length": faces_length, "errors": errors})



@app.route("/api/align", methods=["POST"])
def align_image():
    uploaded_images = request.form.getlist("images")
    faces_length = [0, 0]
    messages = []
    errors = []
    images = []
    for i in range(len(uploaded_images)):
        filename = uploaded_images[i]
        if "aligned" in filename or "detected" in filename:
            path = os.path.join(STATIC_FOLDER, filename)
        else:
            path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(path):
            face_count = helper.create_aligned_images(filename, images)
            faces_length[i]=face_count;
            messages.append(f"{face_count} detected faces in {filename}.")
        else:
            errors.append(f"File {filename} does not exist!")
    return jsonify(
        {
            "images": images,
            "faces_length": faces_length,
            "errors": errors,
            "messages": messages,
        }
    )


@app.route("/api/detect", methods=["POST"])
def detect_image():
    uploaded_images = request.form.getlist("images")
    faces_length = [0, 0]
    messages = []
    errors = []
    images = []
    for i in range(len(uploaded_images)):
        filename = uploaded_images[i]
        if "aligned" in filename or "detected" in filename:
            path = os.path.join(STATIC_FOLDER, filename)
        else:
            path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(path):
            face_count = helper.detect_faces_in_image(
                filename, images
                )
            faces_length[i]=face_count;
            messages.append(f"{face_count} detected faces in {filename}.")
        else:
            errors.append(f"File {filename} does not exist!")
    return jsonify(
        {
            "images": images,
            "faces_length": faces_length,
            "errors": errors,
            "messages": messages,
        }
    )


@app.route("/api/compare", methods=["POST"])
def compare_image():
    uploaded_images = request.form.getlist("images");
    combochanges=[int(x) for x in request.form.getlist("selected_faces")];
    embeddings = []
    messages = []
    errors = []
    image_paths = [
        os.path.join(UPLOAD_FOLDER, filename) for filename in uploaded_images
    ]
    for i in range(len(uploaded_images)):
        if len(uploaded_images) == 2:
            filename=f"aligned_{0 if combochanges[i] == -2 else combochanges[i]}_{uploaded_images[i]}"
            embedding=manager.get_embedding_by_name(filename)
            if len(embedding)>0:
                embeddings.append(embedding);
            else:
                # Generate an embedding for a specific face(first by default) in each image
                embedding, temp_err = helper.generate_embedding(
                    image_paths[i], combochanges[i]
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
        similarity = ImageHelper.calculate_similarity(
            embeddings[0], embeddings[1]
        )
        messages.append(f"Similarity: {similarity:.4f}")
        if similarity >= 0.6518:
            messages.append("THIS IS PROBABLY THE SAME PERSON")
        else:
            messages.append("THIS IS PROBABLY NOT THE SAME PERSON")

    elif len(uploaded_images) != 2:
        errors.append("choose 2 images!")
    else:
        errors.append("Error: Failed to extract embeddings from images.")
    
    return jsonify(
            {
                "errors": errors,
                "messages": messages,
            });

@app.route("/api/check", methods=["POST"])
# def find_similar_image():
#  check_uploaded_images = session.get('check_uploaded_images', [])
#  selected_face = int(request.form.get('selected_face', -2))
#  check_uploaded_images = [x for x in check_uploaded_images if x != '']
#  uploaded_images=alignforcheck(selected_face,check_uploaded_images[0],images)
#  if len(check_uploaded_images) ==1:
#  #embedder = load_model_for_embedding()
#   user_image_path = os.path.join(app.config['UPLOAD_FOLDER'], check_uploaded_images[0])
#   user_img = cv2.imread(user_image_path)
#   user_faces = embedder.get(user_img)
#   user_embedding = extract_embedding(embedder, user_faces[0])
#   max_similarity = -1
#   with os.scandir(UPLOAD_FOLDER) as entries:
#    for entry in entries:
#       if  entry.name != check_uploaded_images[0] and check_uploaded_images[0] not in entry.name:
#         if check_uploaded_images[0] not in entry.name:
#           if entry.is_file()  and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff')):
#              with Image.open(entry.path) as img:
#                if embedder:
#                  img = cv2.imread(entry.path)
#                  faces = embedder.get(img)
#                  if faces:
#                   embedding = extract_embedding(embedder, faces[0])
#                   if embedding is not None:
#                     similarity = calculate_similarity(user_embedding, embedding)
#                     if similarity > max_similarity:
#                       max_similarity = similarity
#                       most_similar_image = entry.name
#  if most_similar_image:
#   # uploaded_images.append(most_similar_image)
#    uploaded_images=alignforcheck(selected_face,most_similar_image,images)
#    message = f"The most similar image is {most_similar_image} with similarity of {max_similarity:.4f}"

def find_similar_image():
    most_similar_image = None
    messages=[];
    errors=[];
    similarity = -1
    face_length=0;
    current_image = request.form.get("image");
    selected_face=int(request.form.get("selected_face"));
    fullfilename = os.path.join(UPLOAD_FOLDER, current_image)
    if len(current_image) > 0  :
        (
            most_similar_image,
            most_similar_face_num,
            similarity,
            temp_err,
        # ) = helper.get_most_similar_image(selected_face, current_image)
        ) = helper.get_most_similar_image_new(selected_face, current_image,fullfilename)
        errors = errors + temp_err
    else:
        errors.append("no images selected for check")
    if most_similar_image:
        messages.append(
            f"The most similar face is no. {most_similar_face_num+1} in image {most_similar_image} with similarity of {similarity:.4f}"
        )
        face_length = helper.create_aligned_images(most_similar_image, [])
    return jsonify(
    {
        "image":most_similar_image,
        "face":most_similar_face_num,
        "face_length":face_length,
        "errors": errors,
        "messages": messages,
    });

app.secret_key = "your_secret_key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ROOT_FOLDER"] = APP_DIR

detector = ModelLoader.load_detector()
embedder = ModelLoader.load_embedder()
manager =ImageEmbeddingManager();
helper = ImageHelper(detector, embedder,manager, UPLOAD_FOLDER, STATIC_FOLDER)


if __name__ == "__main__":
    try:
        app.run(debug=True, port=5057)
    except Exception as e:
        print(f"Error: {e}")
