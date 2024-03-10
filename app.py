from flask import Flask, request, session, jsonify
import os
import cv2
import numpy as np
import tempfile
from modules import ImageEmbeddingManager,ImageHelper,ImageGroupRepository,FamilyClassifier,ModelLoader,util
from flask_cors import CORS
from insightface.utils.face_align import norm_crop
from flask import send_from_directory
import json
import traceback

APP_DIR = os.path.dirname(__file__)
from PIL import Image

UPLOAD_FOLDER = os.path.join(APP_DIR, "pool")
STATIC_FOLDER = os.path.join(APP_DIR, "static")

# Create the folders if they don't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

def custom_sort(filename):
     return (not filename.startswith('detected'), filename)
def sort_detected():
    sorted_files = sorted(os.listdir(UPLOAD_FOLDER), key=custom_sort)
    for filename in sorted_files:
    # Check if the filename starts with 'detected'
     if filename.startswith('detected'):
        file_path = os.path.join(UPLOAD_FOLDER, filename)
        try:
            os.remove(file_path)
            print(f"Deleted: {filename}")
        except Exception as e:
            print(f"Failed to delete {filename}. Reason: {e}")
     else:
        break




def alignforcheck(selected_face, filename, images):
    path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    img = cv2.imread(path)
    faces = detector.get(img)
    for face in faces:
        landmarks = face["kps"].astype(np.int)
        aligned_filename = f"aligned_{selected_face}_{filename}"  # Name contains the selected face index
        aligned_path = os.path.join(STATIC_FOLDER, aligned_filename)
        aligned_img = norm_crop(img, landmarks, 112, "arcface")
        cv2.imwrite(aligned_path, aligned_img)
        images.append(aligned_filename)
        uploaded_images = images
        return uploaded_images


def extract_landmark_features(face):
    # Convert keypoints to integers and flatten the array
    landmarks_array = face["kps"].astype(int).flatten()

    # Convert the NumPy array to a string in the desired format
    landmarks_string = "[" + " ".join(map(str, landmarks_array)) + "]"

    return landmarks_string


def format_landmarks_as_string(landmarks):
    # Check if landmarks is already a string
    if isinstance(landmarks, str):
        return landmarks

    # Check if landmarks is a list with one element which is a string
    if (
        isinstance(landmarks, list)
        and len(landmarks) == 1
        and isinstance(landmarks[0], str)
    ):
        return landmarks[0]

    # Convert list or array of numbers into the desired string format
    if isinstance(landmarks, list) or hasattr(landmarks, "flatten"):
        # Flatten the array if it's not already flat
        if hasattr(landmarks, "flatten"):
            landmarks = landmarks.flatten()

        # Convert landmarks to integers, then to strings, and join with space
        landmarks_string = " ".join(map(str, map(int, landmarks)))

        return f"[{landmarks_string}]"

    # If landmarks is none of the above, return it as is or handle the error
    return landmarks




app = Flask(__name__, static_folder=STATIC_FOLDER)

cors = CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins
app.config["CORS_HEADERS"] = "Content-Type"


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
    if "uploaded_images" not in session:
        session["uploaded_images"] = []

    session["uploaded_images"].append(filename)
    session.modified = True

    return True, "Image uploaded and validated successfully."


@app.route("/api/upload", methods=["POST"])
def upload_image():
    errors = []
    current_images = []
    files = request.files.items()
    faces_length = [0] * len(request.files)
    valid_images=[]
    generated=[];
    invalid_images=[]
    for image_name, file in files:
        if file and file.filename:
            filename = file.filename.replace("_", "")
            if ImageHelper.allowed_file(file.filename):
                session.pop("uploaded_images", None)
                path = os.path.join(UPLOAD_FOLDER, file.filename)
                try:
                    file.save(path)
                    # Generate the embeddings for all faces and store them for future indexing
                    _, temp_err = helper.generate_all_emb(file.filename)
                    errors = errors + temp_err

                    if len(temp_err) > 0:
                        # os.remove(path)
                        invalid_images.append(file.filename);
                        current_images.append(None)
                    else:
                        current_images.append(file.filename)
                        valid_images.append(file.filename)
                except Exception as e:
                    tb = traceback.format_exc()
                    errors.append(f"Failed to save {filename} due to error: {str(e)}")

            else:
                errors.append(f"Invalid file format for {file.filename}. ")

    # if len(errors) == 0:
        # Detect faces immediately after uploading to fill the combo box
    for i in range(len(valid_images)):
            faces_length[i] = helper.create_aligned_images(
                valid_images[i], generated
            )

    manager.save()
    return jsonify({"images": valid_images+invalid_images, "faces_length": faces_length, "errors": errors})


@app.route("/api/delete", methods=["GET"])
def delete_embeddings():
    manager.delete()
    return jsonify({"result": "success"})


# @app.route("/api/clear", methods=["GET"])
# def clear_image():
#     #current_detected_images = []
#     uploaded_images = request.form.getlist("images")
#     checked_images1 = request.form.get("checked_images1")
#     checked_images2 = request.form.get("checked_images2")
#     faces_length = []
#     messages = []
#     errors = []
#     images = []
#     return jsonify(
#         {
#             "images": images,
#             "faces_length": faces_length,
#             "errors": errors,
#             "messages": messages,
#         }
#     )



@app.route("/api/align", methods=["POST"])
def align_image():
    uploaded_images = request.form.getlist("images")
    faces_length = []
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
            faces_length.append(face_count)
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
    #current_detected_images = []
    uploaded_images = request.form.getlist("images")
    faces_length = []
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
            face_count, _ = helper.detect_faces_in_image(filename, images)
            detected_filename = "detected_" + filename
            # detected_path = os.path.join(UPLOAD_FOLDER, detected_filename)

            # os.remove(detected_path)
             


            detected_path = os.path.join(STATIC_FOLDER, detected_filename)
            img = cv2.imread(path)
            cv2.imwrite(detected_path, img)
            
            if face_count is not None:
                faces_length.append(face_count)
                messages.append(f"{face_count} detected faces in {filename}.")
            else:
                errors.append(
                    f"you can't detect this File {filename} beacuse its  does not contain a face!"
                )
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
    sort_detected()
    uploaded_images = request.form.getlist("images")
    combochanges = [int(x) for x in request.form.getlist("selected_faces")]
    embeddings = []
    messages = []
    errors = []
    image_paths = [
        os.path.join(UPLOAD_FOLDER, filename) for filename in uploaded_images
    ]
    for i in range(len(uploaded_images)):
        if len(uploaded_images) == 2:
            filename = f"aligned_{0 if combochanges[i] == -2 else combochanges[i]}_{uploaded_images[i]}"
            embedding = manager.get_embedding_by_name(filename)
            if len(embedding) > 0:
                embeddings.append(embedding)
            else:
                embedding, temp_err = helper.generate_embedding(
                    uploaded_images[i], combochanges[i]
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
        }
    )


@app.route("/api/improve", methods=["POST"])
def improve_image():
   
    uploaded_images = request.form.getlist("images")
    checked_images1 = request.form.get("checked_images1")
    checked_images2 = request.form.get("checked_images2")
    enhanced_images = []
    errors = []
    messages=[]
    i=0
    options=[]
    print(request.form)
    for filename in uploaded_images:
        i=i+1
        if "enhanced" not in filename:
            try:
                if(i==1):
                    if(checked_images1=='True'):
                      enhanced_img = helper.enhance_image(filename)
                      if enhanced_img is not None:
                        enhanced_image_path = os.path.join(UPLOAD_FOLDER,"enhanced_"+filename)
                        cv2.imwrite(enhanced_image_path, enhanced_img)
                        enhanced_images.append("enhanced_"+filename)
                    else:
                         enhanced_images.append(filename);
                else:
                     if(checked_images2=='True'):
                         enhanced_img = helper.enhance_image(filename)
                         if enhanced_img is not None:
                          enhanced_image_path = os.path.join(UPLOAD_FOLDER,"enhanced_"+filename)
                          cv2.imwrite(enhanced_image_path, enhanced_img)
                          enhanced_images.append("enhanced_"+filename)
                     else:
                         enhanced_images.append(filename);
                         
                         
                         

            except Exception as e:
                errors.append(str(e))
        else:
             if(i==1):
                  if(checked_images1=='True'):
                    filename = filename.replace("enhanced_", "")
                    enhanced_images.append(filename);
                  else:
                      enhanced_images.append(filename);
             else:
                  if(checked_images2=='True'):
                    filename = filename.replace("enhanced_", "")
                    enhanced_images.append(filename);
                  else:
                      enhanced_images.append(filename);
                 
                      
                 
            
            #messages.append(f"check")
            #options.append(messages.option)
    
            #errors.append(f"image {filename} is already enhanced!")
    return jsonify(
        {
            "enhanced_images": enhanced_images,
            "errors": errors,
            "messages":messages,
            "options": ["1", "2", "all"],

        }
    )


@app.route("/api/check_family", methods=["POST"])
def checkisfamily():
    
# Get a list of all files in the directory and sort them using the custom sort function
    sort_detected()

    uploaded_images = request.form.getlist("images")
    embeddings = []
    messages = []
    errors = []
    if len(uploaded_images) == 2:
        for i in range(len(uploaded_images)):
            #check if first name embedding already exists in repository
            aligned_filename = f"aligned_0_{uploaded_images[i]}"
            embedding = manager.get_embedding_by_name(aligned_filename)
            if len(embedding) > 0:
                embeddings.append(embedding)
            else:
                embedding, temp_err = helper.generate_embedding(
                    uploaded_images[i], 0
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
        classifier= FamilyClassifier(APP_DIR);
        Genders=[0,0]
        # embedding_str_1 = np.array2string(embeddings[0], separator=',', precision=10, max_line_width=np.inf)
        # embedding_str_1_single_line = embedding_str_1.replace('\n', ' ').replace('[', '').replace(']', '')
        # embedding_list_1 = [float(item) for item in embedding_str_1_single_line.split(',') if item.strip()]
        # embedding1 = np.array(embedding_list_1)
        # print(embedding1)

        # embedding_str_2 = np.array2string(embeddings[1], separator=',', precision=10, max_line_width=np.inf)
        # embedding_str_2_single_line = embedding_str_2.replace('\n', ' ').replace('[', '').replace(']', '')
        # embedding_list_2 = [float(item) for item in embedding_str_2_single_line.split(',') if item.strip()]
        # embedding2 = np.array(embedding_list_2)
        # for i in range (2):
        

        for i in range(2):
            img,faces=helper._ImageHelper__extract_faces(uploaded_images[i]);
            if faces:
                Genders[i]=faces[0].gender;
       
        #  if(i==0):
        #   landmarks1.append(extract_landmark_features(faces[0]))
        #   formatted_landmarks1 = format_landmarks_as_string(landmarks1)
        #  else:
        #   landmarks2.append(extract_landmark_features(faces[0]))
        #   formatted_landmarks2 = format_landmarks_as_string(landmarks2)
        # landmarks1= "[16. 19. 48. 18. 33. 33. 19. 42. 46. 41.]"

        # landmarks2="[18. 21. 47. 21. 32. 39. 20. 44. 47. 44.]"

        is_same_family=classifier.predict(similarity,Genders[0],Genders[1]);
        messages.append(
            "Probably the Same Family"
            if is_same_family
            else "Probably Not the Same Family"
        )

    return jsonify(
        {
            "errors": errors,
            "messages": messages,
        }
    )


@app.route("/api/check_many", methods=["POST"])
def find_similar_images():
    sort_detected()
    errors = []
    images = []
    files = request.files.items()
    k = int(request.form.get("number_of_images", 5))
    with tempfile.TemporaryDirectory() as temp_dir:
        for image_name, file in files:
            if file and file.filename:
                filename = file.filename.replace("_", "")
                if ImageHelper.allowed_file(filename):
                    path = os.path.join(temp_dir, filename)
                    try:
                        file.save(path)
                        # Generate the embeddings for all faces and store them for future indexing
                        embs, temp_err = helper.generate_all_emb(filename, False)
                        similar = helper.get_similar_images(embs[0], filename, k)
                        for x in similar:
                            sim_emb = manager.get_embedding(x["index"])
                            similarity = util.calculate_similarity(embs[0], sim_emb)
                            images.append(
                                {"name": x["name"], "similarity": float(similarity)}
                            )

                        errors = errors + temp_err

                    except Exception as e:
                        errors.append(
                            f"Failed to save {filename} due to error: {str(e)}"
                        )

                else:
                    errors.append(f"Invalid file format for {filename}. ")
    return jsonify({"images": images, "errors": errors})


@app.route("/api/check", methods=["POST"])
def find_similar_image():
    sort_detected()
    most_similar_image = None
    messages = []
    errors = []
    similarity = -1
    face_length = 0
    current_image = request.form.get("image")
    selected_face = int(request.form.get("selected_face"))
    fullfilename = os.path.join(UPLOAD_FOLDER, current_image)
    if len(current_image) > 0:
        (
            most_similar_image,
            most_similar_face_num,
            similarity,
            temp_err,
            ) = helper.get_most_similar_image(selected_face, current_image)
        # ) = helper.get_most_similar_image_new(
        #     selected_face, current_image, fullfilename
        # )
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
            "image": most_similar_image,
            "face": most_similar_face_num,
            "face_length": face_length,
            "errors": errors,
            "messages": messages,
        }
    )


@app.route("/api/find", methods=["POST"])
def find_face_in_image():
    sort_detected()
    filename = request.form.get("image")
    faces_length = [0]
    messages = []
    errors = []
    boxes = []
    if "aligned" in filename or "detected" in filename:
        path = os.path.join(STATIC_FOLDER, filename)
    else:
        path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        face_count, boxes = helper.detect_faces_in_image(filename, [])
        faces_length = face_count
        messages.append(f"{face_count} detected faces in {filename}.")
    else:
        errors.append(f"File {filename} does not exist!")
    return jsonify(
        {
            "boxes": boxes,
            "faces_length": faces_length,
            "errors": errors,
            "messages": messages,
        }
    )


@app.route("/api/filter",methods=["POST"])
def filter():
    threshold = float(request.form.get("threshold", 999))
    if threshold<1:
        deleted=helper.filter(threshold)
        return jsonify({"success":True,"deleted":deleted})
    return jsonify({"success":False,"error":"threshold must be below 1"})

@app.route("/api/ping",methods=["GET"])
def ping():
    return jsonify({"response":"pong"});



@app.route("/api/cluster", methods=["POST"])
def get_groups():
    jsonData=request.get_data();
    data=json.loads(jsonData) if jsonData else {};
    eps=float(data["max_distance"]) if "max_distance" in data else 0.5; 
    min_samples=int(data["min_samples"]) if "min_samples" in data else 4; 
    retrain=data["retrain"] if "retrain" in data else False; 
    value_groups=helper.cluster_images(eps,min_samples);
    if(retrain):
        groups.train_index(value_groups);
        groups.save_index();
        return jsonify(value_groups);
    modified_group={};
    index=groups.index;
    for cluster_id,images in value_groups.items():
        for image in images:
            group_name=cluster_id;
            if(image in index):
                group_name=index[image];
            if(group_name in modified_group):
                modified_group[group_name].append(image);
            else:
                modified_group[group_name]=[image];
    return jsonify(modified_group);

@app.route("/api/video",methods=["POST"])
def process_video():
    pass;


@app.route("/api/change_group_name", methods=["POST"])
def change_group_name():
    data=json.loads(request.get_data());
    old= data['old'];
    new= data['new'];
    if(old and new and old.strip() and new.strip()):
        groups.change_group_name(old,new);
    return jsonify(success=True);

app.secret_key = "your_secret_key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ROOT_FOLDER"] = APP_DIR


detector = ModelLoader.load_detector(1024)
detector_zoomed = ModelLoader.load_detector(160)
embedder = ModelLoader.load_embedder(64)
manager = ImageEmbeddingManager(APP_DIR)
groups = ImageGroupRepository()
helper = ImageHelper(
    detector, detector_zoomed, embedder, groups, manager, UPLOAD_FOLDER, STATIC_FOLDER
)
manager.load()

if __name__ == "__main__":
    try:
        app.run(debug=True, port=5057)
    except Exception as e:
        print(f"Error: {e}")
