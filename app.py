from flask import Flask, request, session, jsonify
import os
import cv2
import sys
from modules import ImageEmbeddingManager,ImageHelper,ImageGroupRepository,FamilyClassifier,ModelLoader,util
from flask_cors import CORS
from insightface.utils.face_align import norm_crop
from flask import send_from_directory
import json
import traceback
APP_DIR = os.path.dirname(sys.argv[0])
UPLOAD_FOLDER = os.path.join(APP_DIR, "pool")
STATIC_FOLDER = os.path.join(APP_DIR,"static")

# create dirs
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.join(UPLOAD_FOLDER,"no_face"), exist_ok=True)
os.makedirs(STATIC_FOLDER, exist_ok=True)

for model in ModelLoader.models:
    os.makedirs(os.path.join(STATIC_FOLDER,model), exist_ok=True)
    


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




app = Flask(__name__, static_folder="static")

cors = CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins
app.config["CORS_HEADERS"] = "Content-Type"


@app.route("/pool/<path:filename>")
def custom_static(filename):
    return send_from_directory("pool", filename)


@app.route("/static_images/<path:filename>")
def processed_static(filename):
    return send_from_directory("static", filename)


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
    faces_length = []
    valid_images=[]
    generated=[];
    #call the buffalo model as a def
    return_model_name=request.form.get("model_name","buffalo_l",type=str)
    save_invalid=request.form.get("save_invalid",False,type=bool);
    invalid_images=[]
    for image_name, file in files:
        if file and file.filename:
            filename = file.filename.replace("_", "")
            if ImageHelper.allowed_file(file.filename):
                session.pop("uploaded_images", None)
                path = os.path.join(UPLOAD_FOLDER, file.filename)
                try:
                    file.save(path)
                    for model_name,_ in ModelLoader.models.items():
                        model=ModelLoader.load_model(model_name=model_name);
                    # Generate the embeddings for all faces and store them for future indexing
                        img,faces,temp_err=helper.create_aligned_images(file.filename,model,generated)
                        img,faces,_,temp_err = helper.generate_all_emb(img,faces,file.filename,model)
                        errors = errors + temp_err
                    #check the image contains a face 
                        if(model_name==return_model_name):
                            faces_length.append(len(faces)) if faces else faces_length.append(0);
                            if len(temp_err) > 0 or (not faces):
                                if(save_invalid):
                                    os.replace(path,os.path.join(UPLOAD_FOLDER,"no_face",file.filename))
                                else:
                                    os.remove(path);
                                invalid_images.append("no_face/"+file.filename);
                                current_images.append(None)
                            else:
                                current_images.append(file.filename)
                                valid_images.append(file.filename)

                        manager.save(model_name)
                except Exception as e:
                    tb = traceback.format_exc()
                    errors.append(f"Failed to save {filename} due to error: {str(e)}")

            else:
                errors.append(f"Invalid file format for {file.filename}. ")

    return jsonify({"images": valid_images,"invalid_images":invalid_images, "faces_length": faces_length, "errors": errors})


@app.route("/api/delete", methods=["GET"])
def delete_embeddings():
    manager.delete_all()
    return jsonify({"result": "success"})

@app.route("/api/align", methods=["POST"])
def align_image():
    uploaded_images = request.form.getlist("images")
    model_name=request.form.get("model_name","buffalo_l",type=str)
    model=ModelLoader.load_model(model_name)
    faces_length = []
    messages = []
    errors = []
    images = []
    # align the images if they aren't already aligned
    for i in range(len(uploaded_images)):
        filename = uploaded_images[i]
        if "aligned" in filename or "detected" in filename:
            path = os.path.join(STATIC_FOLDER, filename)
        else:
            path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(path):
            _,faces = helper.create_aligned_images(filename,model, images)
            faces_length.append(len(faces))
            messages.append(f"{len(faces)} detected faces in {filename}.")
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
    model_name=request.form.get("model_name","buffalo_l",type=str)
    model=ModelLoader.load_model(model_name)
    faces_length = []
    messages = []
    errors = []
    images = []
    for i in range(len(uploaded_images)):
        filename = uploaded_images[i]
        if "aligned" in filename or "detected" in filename:
            path = os.path.join(STATIC_FOLDER,model_name, filename)
        else:
            path = os.path.join(UPLOAD_FOLDER, filename)
        if os.path.exists(path):
            # ensure that the user check img with faces for the detection
            face_count, _ = helper.detect_faces_in_image(filename,model, images)
            
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
    uploaded_images = request.form.getlist("images")
    similarity_thresh=request.form.get("similarity_thresh",0.5,type=float)
    model_name=request.form.get("model_name","buffalo_l",type=str)
    model=ModelLoader.load_model(model_name)
    combochanges = [int(x) for x in request.form.getlist("selected_faces")]
    embeddings = []
    messages = []
    errors = []
    # ensure that the user check 2 image faces 
    for i in range(len(uploaded_images)):
        if len(uploaded_images) == 2:
            filename = f"aligned_{0 if combochanges[i] == -2 else combochanges[i]}_{uploaded_images[i]}"
            embedding = manager.get_embedding_by_name(filename,model_name).embedding
            if len(embedding) > 0:
                embeddings.append(embedding)
            else:
                embedding, temp_err = helper.generate_embedding(
                    uploaded_images[i], combochanges[i],model
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
        if similarity >= similarity_thresh:
            messages.append(f"THIS IS PROBABLY THE SAME PERSON ")
        else:
            messages.append(f"THIS IS PROBABLY NOT THE SAME PERSON")

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
   
    image = request.form.get("image")
    model_name=request.form.get("model_name","buffalo_l",type=str)
    model=ModelLoader.load_model(model_name)

    enhanced_image=image
    errors = []
    messages=[]
    i=0
    print(request.form)
    #check if the image is already enhanced
    if "enhanced" not in image:
        try:
            enhanced_img = helper.enhance_image(image)
            if enhanced_img is not None:
                enhanced_image_path = os.path.join(UPLOAD_FOLDER,"enhanced_"+image)
                cv2.imwrite(enhanced_image_path, enhanced_img)
                enhanced_image="enhanced_"+image

                img,faces,temp_err=helper.create_aligned_images("enhanced_"+image,model,[])
                _,_,_,temp_err = helper.generate_all_emb(img,faces,"enhanced_"+image,model)
                errors = errors + temp_err

                if len(temp_err) > 0:
                    os.remove(enhanced_image_path);
        except Exception as e:
            errors.append(str(e))
    else:
        errors.append(f"image {image} is already enhanced!")
    return jsonify(
        {
            "enhanced_image": enhanced_image,
            "errors": errors,
            "messages":messages,
        }
    )


@app.route("/api/check_family", methods=["POST"])
def checkisfamily():
    model_name=request.form.get("model_name","buffalo_l",type=str)
    #similarity_thresh=request.form.get("similarity_thresh",0.6,type=float)
    model=ModelLoader.load_model(model_name)
    uploaded_images = request.form.getlist("images")
    # helper.cluster_family_images(model_name,APP_DIR,uploaded_images,model)
    combochanges = [int(x) for x in request.form.getlist("selected_faces")]
    embeddings = []
    messages = []
    errors = []
    #check if the user upload 2 face images
    if len(uploaded_images) == 2:
        for i in range(len(uploaded_images)):
            #check if first name embedding already exists in repository
            aligned_filename = f"aligned_{0 if combochanges[i] == -2 else combochanges[i]}_{uploaded_images[i]}"
            embedding = manager.get_embedding_by_name(aligned_filename,model_name).embedding
            if len(embedding) > 0:
                embeddings.append(embedding)
            else:
                embedding, temp_err = helper.generate_embedding(
                    uploaded_images[i], 0,model
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
        
        #get the min of each face 
        for i in range(2):
            img,faces=helper._ImageHelper__extract_faces(uploaded_images[i],model);
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

        is_same_family,similar=classifier.predict(similarity,Genders[0],Genders[1]);
        # probability=classifier.predict(similarity,Genders[0],Genders[1]);
        # is_same_family= probability>similarity_thresh
        messages.append(
            f"Probably the Same Family with probability {similar:.2f}"
            if is_same_family
            else f"Probably Not the Same Family with probability {similar:.2f}"
        )

    return jsonify(
        {
            "errors": errors,
            "messages": messages,
        }
    )


@app.route("/api/check_many", methods=["POST"])
def find_similar_images():
    errors = []
    messages = []
    model_name=request.form.get("model_name","buffalo_l",type=str)
    model=ModelLoader.load_model(model_name)
    similarity_thresh=request.form.get("similarity_thresh",0.5,type=float)
    current_image = request.form.get("image")
    selected_face = int(request.form.get("selected_face"))
    k = int(request.form.get("number_of_images", 5))
    similar,temp_err = helper.get_k_similar_images(current_image,selected_face,similarity_thresh,model, k)
    errors = errors + temp_err
    json_similar=[s.to_json() for s in similar]
    return jsonify({"images": json_similar, "errors": errors})


@app.route("/api/check", methods=["POST"])
def find_similar_image():
    model_name=request.form.get("model_name","buffalo_l",type=str)
    similarity_thresh=request.form.get("similarity_thresh",0.5,type=float)
    model=ModelLoader.load_model(model_name)

    most_similar_image = None
    messages = []
    errors = []
    image_name=""
    face_num=-2
    face_length = 0
    current_image = request.form.get("image")
    selected_face = int(request.form.get("selected_face"))
    if len(current_image) > 0:
       
        similar_images,temp_err=helper.get_k_similar_images(current_image,
                                                            selected_face,
                                                            similarity_thresh,
                                                            model)
        errors = errors + temp_err
        if len(similar_images)>0:
            most_similar_image=max(similar_images,key=lambda x: x.similarity)
    else:
        errors.append("no images selected for check")
    if most_similar_image:
        messages.append(
            f"The most similar face is no. {most_similar_image.face_num+1} in image {most_similar_image.image_name} with similarity of {most_similar_image.similarity:.4f}"
        )
        face_length = len(helper.get_face_boxes(most_similar_image.image_name,model_name))
        image_name=most_similar_image.image_name
        face_num=most_similar_image.face_num
    return jsonify(
        {
            "image": image_name,
            "face": face_num,
            "face_length": face_length,
            "errors": errors,
            "messages": messages,
        }
    )

@app.route("/api/check_template", methods=["POST"])
def find_by_template():
    most_similar_image = None
    messages = []
    errors = []
    box=[]
    similarity = -1
    current_image = request.form.get("template")
    similarity_thresh=request.form.get("similarity_thresh",0.5,type=float)
    if len(current_image) > 0:
        (
            most_similar_image,
            box,
            similarity,
            temp_err,
            ) = helper.get_most_similar_image_by_template(current_image,similarity_thresh)
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

@app.route("/api/find", methods=["POST"])
def find_face_in_image():
    filename = request.form.get("image")
    model_name=request.form.get("model_name","buffalo_l",type=str)
    model=ModelLoader.load_model(model_name)
    faces_length = [0]
    messages = []
    errors = []
    boxes = []
    if "aligned" in filename or "detected" in filename:
        path = os.path.join(STATIC_FOLDER, filename)
    else:
        path = os.path.join(UPLOAD_FOLDER, filename)
    if os.path.exists(path):
        boxes = helper.get_face_boxes(filename,model.name)
        faces_length = len(boxes)
        messages.append(f"{faces_length} detected faces in {filename}.")
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
    model_name=request.form.get("model_name","buffalo_l",type=str)
    threshold = float(request.form.get("threshold", 999))
    if threshold<1:
        deleted=helper.filter(threshold,model_name=model_name)
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
    model_name=data["model_name"] if "model_name" in data else "buffalo_l"
    cluster_family = data.get("cluster_family", False)
    #for now- if it's the same family its change the threshold to 0.13
    if (cluster_family):
      eps=0.87  
    value_groups=helper.cluster_images(eps,min_samples,model_name);
    
   
    if(retrain):
        groups.train_index(value_groups,model_name);
        groups.save_index(model_name);
        return jsonify(value_groups);
    modified_group={};
    index=groups.groups[model_name].index;
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


@app.route("/api/change_group_name", methods=["POST"])
def change_group_name():
    data=json.loads(request.get_data());
    old= data['old'];
    new= data['new'];
    model_name=data["model_name"] if "model_name" in data else "buffalo_l"
    if(old and new and old.strip() and new.strip()):
        groups.change_group_name(old,new,model_name);
    return jsonify(success=True);

app.secret_key = "your_secret_key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["ROOT_FOLDER"] = APP_DIR

manager = ImageEmbeddingManager(APP_DIR)
groups = ImageGroupRepository(APP_DIR)
helper = ImageHelper(
    groups, manager, UPLOAD_FOLDER, STATIC_FOLDER
)
for model_name,_ in ModelLoader.models.items():
    ModelLoader.load_model(model_name,APP_DIR)
    manager.load(model_name)

if __name__ == "__main__":
    try:
        app.run(debug=True, port=5057,use_reloader=False)
    except Exception as e:
        print(f"Error: {e}")