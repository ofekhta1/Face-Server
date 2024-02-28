import numpy as np
import pandas as pd
from insightface.utils.face_align import norm_crop
import os
from model_loader import ModelLoader
from joblib import load
import cv2
import pickle
from PIL import Image
from image_embedding_manager import ImageEmbeddingManager
class ImageHelper:
  
    ALLOWED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp"
    }
    @staticmethod
    def string_to_numpy_array(string):
        # Removing brackets and splitting the string
        str_values = string.replace('[', '').replace(']', '').split()
        # Converting each string to a float
        float_values = [float(val) for val in str_values]
        # Converting the list of floats to a numpy array
        return np.array(float_values)
    @staticmethod
    def create_features(df):
     features = []
     for _, row in df.iterrows():
        gender_pic1_numeric =ImageHelper.gender_to_numeric(row['Gender1'])
        gender_pic2_numeric =ImageHelper.gender_to_numeric(row['Gender2'])
        similarity_score = row['similatrity']  # Make sure this column name is correct
        # landmarks1 = row['landmarks1'].flatten()  # Flatten the landmarks array
        # landmarks2 = row['landmarks2'].flatten()  # Flatten the landmarks array

        # Combine the features into a single feature array
        feature = np.concatenate(([gender_pic1_numeric, gender_pic2_numeric, similarity_score],))
        features.append(feature)
    
     return np.array(features)
    @staticmethod
    def gender_to_numeric(gender):
    # Convert gender to numeric (e.g., 'M' to 0 and 'W' to 1)
     return 1 if gender == 'W' else 0

    # Load model on startup
    def __init__(self, detector,embedder,emb_manager, UPLOAD_FOLDER, STATIC_FOLDER):
        self.detector = detector
        self.embedder = embedder
        self.UPLOAD_FOLDER = UPLOAD_FOLDER
        self.STATIC_FOLDER = STATIC_FOLDER
        self.emb_manager=emb_manager;
    @staticmethod
    def points(numpoints,max_val,template_path,image_path):
      MIN_MATCH_COUNT = numpoints
      img1 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) # queryImage
      img2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # trainImage
      sift = cv2.SIFT_create()
      kp1, des1 = sift.detectAndCompute(img1,None)
      kp2, des2 = sift.detectAndCompute(img2,None)
      FLANN_INDEX_KDTREE = 1
      index_params = dict(algorithm = FLANN_INDEX_KDTREE, trees = 5)
      search_params = dict(checks = 50)
      flann = cv2.FlannBasedMatcher(index_params, search_params)
      matches = flann.knnMatch(des1,des2,k=2)
      good = []
      M = 0
      for m,n in matches:
       if m.distance < 0.7*n.distance:
        good.append(m)
       if (len(good)>MIN_MATCH_COUNT) and (max_val>= - 0.2 or len(good)>=60):
         src_pts = np.float32([ kp1[m.queryIdx].pt for m in good ]).reshape(-1,1,2)
         dst_pts = np.float32([ kp2[m.trainIdx].pt for m in good ]).reshape(-1,1,2)
         M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC,5.0)
         matchesMask = mask.ravel().tolist()
         h,w = img1.shape
         pts = np.float32([ [0,0],[0,h-1],[w-1,h-1],[w-1,0] ]).reshape(-1,1,2)
         if(M is not None):
          dst = cv2.perspectiveTransform(pts,M)
          img2 = cv2.polylines(img2,[np.int32(dst)],True,255,3, cv2.LINE_AA)
       else:
         #print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
         matchesMask = None
      draw_params = dict(matchColor = (0,255,0), # draw matches in green color
      singlePointColor = None,
      matchesMask = matchesMask, # draw only inliers
      flags = 2)
      img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
      if (len(good) >= MIN_MATCH_COUNT) and (max_val>= - 0.2 or len(good)>=60) and M is not None :
        print("The object (e.g., tattoo) exists in both images!")
        #plt.imshow(img3, 'gray')
        #plt.show(block=True)
        #create_combined_file()
      else:
     #plt.imshow(img3, 'gray')
     #plt.show(block=True)
         print("The object (e.g., tattoo) does NOT exist or the similarity is too low.")
      if(M is not None):
            return len(good)
      else:
          return 0

    

     
    def create_combined_file(max_loc,image,template):
        h, w = image.shape[:2]
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        detected_object = template[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
        detected_object_resized = cv2.resize(detected_object, (image.shape[1], image.shape[0]))
        combined_image = np.hstack((image, detected_object_resized))

    # output_dir = create_output_directory(image_files, images_dir + r"\archive")
    # combined_image_path = os.path.join(output_dir, "combined_" + image_files)
    # cv2.imwrite(combined_image_path, combined_image)
    @staticmethod
    def string_to_numpy_array(string):
        # Removing brackets and splitting the string
        str_values = string.replace('[', '').replace(']', '').split()
        # Converting each string to a float
        float_values = [float(val) for val in str_values]
        # Converting the list of floats to a numpy array
        return np.array(float_values)
    @staticmethod
    def load_model(model_path):
    # Load the saved model
     return load(model_path)
    @staticmethod
    def predict_family(model,similarity, Gender1, Gender2):
        with open(r'C:\work\python\family_model\scaler.pkl', 'rb') as scaler_file:
         scaler = pickle.load(scaler_file)
        with open(r'C:\work\python\family_model\family_classifier_model.pkl', 'rb') as model_file:
         model = pickle.load(model_file)
        new_data = {
    "Gender1": Gender1,
    "Gender2": Gender2,
    "similatrity": similarity
    # "landmarks1": landmarks1,
    # "landmarks2": landmarks2
     }
        # new_data['landmarks1'] =ImageHelper.string_to_numpy_array(new_data['landmarks1'])
        # new_data['landmarks2'] = ImageHelper.string_to_numpy_array(new_data['landmarks2'])
        features =ImageHelper.create_features(pd.DataFrame([new_data]))
        features_scaled = scaler.transform(features)
        prediction = model.predict(features_scaled)
        return prediction[0]        
     
           
    # Flatten the embeddings if they are 2D
    #  if embedding1.ndim > 1:
    #     embedding1 = embedding1.flatten()
    #  if embedding2.ndim > 1:
    #     embedding2 = embedding2.flatten()

    #  # Convert Gender1 and Gender2 to numpy arrays
    #  Gender1_array = np.array([Gender1])
    #  Gender2_array = np.array([Gender2])

    # # Concatenate all arrays
    #  feature = np.concatenate((embedding1, embedding2, Gender1_array, Gender2_array))

    # # Reshape the feature for prediction
    #  feature = feature.reshape(1, -1)

    # # Make a prediction
    #  prediction = model.predict(feature)
    #  return prediction[0]
    

    @staticmethod
    def calculate_similarity(emb_a, emb_b):
        similarity = np.dot(emb_a, emb_b) / (
            np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
        )
        return similarity

    def __align_single_image(self, face, selected_face, filename, img):
        landmarks = face["kps"].astype(int)
        aligned_filename = f"aligned_{selected_face}_{filename}"
        aligned_path = os.path.join(self.STATIC_FOLDER, aligned_filename)
        aligned_img = norm_crop(img, landmarks, 112, "arcface")
        cv2.imwrite(aligned_path, aligned_img)
        return aligned_filename

    def detect_faces_in_image(self, filename, images):
        img, faces = self.__extract_faces(filename)
        if faces:
            for face in faces:
                landmarks = face["kps"].astype(int)
                for point in landmarks:
                    cv2.circle(
                        img,
                        (int(point[0]), int(point[1])),
                        2,
                        (0, 255, 0),
                        -2,
                    )

            detected_filename = "detected_" + filename
            detected_path = os.path.join(self.STATIC_FOLDER, detected_filename)
            # message += f"path {detected_path}. "
            cv2.imwrite(detected_path, img)
            images.append(detected_filename)
            return len(faces)
        else:
            images.append(filename)
   
     
          

    def create_aligned_images(self, filename, images):
        img, faces = self.__extract_faces(filename)
        face_count = 0

        for face in faces:
            aligned_filename = self.__align_single_image(
                face, face_count, filename, img
            )
            images.append(aligned_filename)
            face_count += 1
        return face_count

    def __extract_faces(self, filename):
        path = os.path.join(self.UPLOAD_FOLDER, filename)
        img = cv2.imread(path)
        faces = self.detector.get(img)
        return img, faces

    @staticmethod
    def extract_embedding(face_data):
        try:
            if face_data and "embedding" in face_data:
                embedding = face_data["embedding"]
                return embedding
            else:
                print("No faces detected.")  # Debug log
                return None
        except Exception as e:
            print("Error during embedding extraction:", e)  # Debug log
            return None
    @staticmethod
    def allowed_file(filename):
        extension=os.path.splitext(filename)[1];
        return extension.lower() in ImageHelper.ALLOWED_EXTENSIONS;
    def generate_all_emb(self,path,filename):
        errors=[];
        embedding=None;        
        if self.embedder:
            img = cv2.imread(path)
            faces = self.embedder.get(img)
            if faces:
                for i in range(len(faces)):
                    embedding = ImageHelper.extract_embedding(faces[i]);
                    self.emb_manager.add_embedding(embedding,f"aligned_{i}_{filename}");
            ## ofek061123
            # else:
            #     print("No faces detected.")  # Debug log
            #     errors.append("No faces detected in one or both images.")
        # else:
        #     errors.append("Error: Embedder model not initialized.")
        return errors;


    def generate_embedding(self,path,selected_face):
        errors=[];
        embedding=None;
        if self.embedder:
            img = cv2.imread(path)
            faces = self.embedder.get(img)
            if faces:
                if selected_face == -2 or len(faces) == 1:
                    i=0
                else:
                    i=selected_face

                embedding = ImageHelper.extract_embedding(faces[i]);
        ##ofek061123
        
            # else:
            #     print("No faces detected.")  # Debug log
            #     errors.append("No faces detected in one or both images.")
        else:
            errors.append("Error: Embedder model not initialized.")
        return embedding,errors;

    def get_most_similar_image(self,selected_face,filename):
        user_image_path = os.path.join(self.UPLOAD_FOLDER, filename)
        errors=[]
        most_similar_image=None;
        max_similarity=-1;
        facenum=-2;
        aligned_filename=f"aligned_{0 if selected_face == -2 else selected_face}_{filename}";
        embedding=self.emb_manager.get_embedding_by_name(aligned_filename)
        if len(embedding)>0:
            user_embedding=embedding;
        else:
            user_embedding,temp_err=self.generate_embedding(user_image_path,selected_face);
            errors=errors+temp_err;
        if len(errors)==0:
            np_emb=np.array(user_embedding).astype("float32").reshape(1,-1)
            idx=self.emb_manager.search(np_emb);
            filtered=[]
            for i in idx:
                name=self.emb_manager.get_name(i);
                if(name.split('_')[-1]!=filename):
                    filtered.append({"index":i,"name":name})
            valid=[x for x in filtered if not np.allclose(self.emb_manager.get_embedding(x['index']),user_embedding,rtol=1e-5,atol=1e-8)]
            if(len(valid)>0):
                match=valid[0]['name'];
                _,facenum,most_similar_image=match.split('_');
                max_similarity=ImageHelper.calculate_similarity(
                    self.emb_manager.get_embedding(valid[0]['index'])
                    ,user_embedding);
            else:
                errors.append("No unique matching faces found!");
        else:
            errors=errors+temp_err;
        return most_similar_image,int(facenum),max_similarity,errors;
    def get_most_similar_image_new(self,selected_face,filename,fullfilename):
        user_image_path = os.path.join(self.UPLOAD_FOLDER, filename)
        errors=[]
        most_similar_image=None;
        max_similarity=-1;
        facenum=-2;
        aligned_filename=f"aligned_{0 if selected_face == -2 else selected_face}_{filename}";
        embedding=self.emb_manager.get_embedding_by_name(aligned_filename)
        if len(embedding)>0:
            user_embedding=embedding;
        else:
            user_embedding,temp_err=self.generate_embedding(user_image_path,selected_face);
            errors=errors+temp_err;
        if len(errors)==0 and (embedding is not None):
            # np_emb=np.array(user_embedding).astype("float32").reshape(1,-1)
            # idx=self.emb_manager.search(np_emb);
            # filtered=[]
            # for i in idx:
            #     name=self.emb_manager.get_name(i);
            #     if(name.split('_')[-1]!=filename):
            #         filtered.append({"index":i,"name":name})
            # valid=[x for x in filtered if not np.allclose(self.emb_manager.get_embedding(x['index']),user_embedding,rtol=1e-5,atol=1e-8)]
            # if(len(valid)>0):
            #     match=valid[0]['name'];
            #     _,facenum,most_similar_image=match.split('_');
            #     max_similarity=ImageHelper.calculate_similarity(
            #         self.emb_manager.get_embedding(valid[0]['index'])
            #         ,user_embedding);
            # else:
            #     errors.append("No unique matching faces found!");
            #   max_similarity = -1
            embedder = ModelLoader.load_embedder(64)
            sum_points=[0]
            with os.scandir(self.UPLOAD_FOLDER) as entries:
                for entry in entries:
                    if  (entry.name != filename) and (filename not in entry.name) and (entry.name != filename.replace("enhanced_", "")):
                        if filename not in entry.name:
                            if entry.is_file()  and entry.name.lower().endswith(('.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tif', '.tiff')):
                                with Image.open(entry.path) as img:
                                    if embedder:
                                        img = cv2.imread(entry.path)
                                        if(img is not None):
                                         faces = embedder.get(img)
                                         if faces:
                                            embedding = ImageHelper.extract_embedding(faces[0]) 
                                            if embedding is not None and user_embedding is not None :
                                                similarity = ImageHelper.calculate_similarity(user_embedding, embedding)
                                                if similarity > max_similarity:
                                                  max_similarity = similarity
                                                  most_similar_image = entry.name
                                            elif(user_embedding is None):
                                               template = cv2.imread(fullfilename) 
                                               image_height, image_width, _ = img.shape
                                               template = cv2.resize(template, (image_width, image_height))
                                               template = template.astype(img.dtype)
                                               result = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
                                               min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                                               threshold = 0.34
                                               if max_val >= threshold:
                                                 h, w, _ = template.shape
                                                 top_left = max_loc
                                                 bottom_right = (top_left[0] + w, top_left[1] + h)
                                                 cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
                                                 most_similar_image = entry.name
                                                 max_similarity =max_val
                                                #create_combined_file() 
                                               elif(max_val>=0.273):
                                                  leng=  ImageHelper.points(4,max_val,fullfilename,entry.path)
                                                  if(leng>=4):
                                                      if max(sum_points) < leng:
                                                       most_similar_image = entry.name
                                                       max_similarity =max_val
                                                      sum_points.append(leng)
                                               elif(max_val>=0.2):
                                                  leng=  ImageHelper.points(5,max_val,fullfilename,entry.path)
                                                  if(leng>=5):
                                                      
                                                      if max(sum_points) < leng:
                                                       most_similar_image = entry.name
                                                       max_similarity =max_val
                                                      sum_points.append(leng) 
                                               else: 
                                                 leng= ImageHelper.points(15,max_val,fullfilename,entry.path)
                                                 if(leng>=15):
                                                      if max(sum_points) < leng:
                                                       most_similar_image = entry.name
                                                       max_similarity =max_val
                                                      sum_points.append(leng)
                                                  
            if(most_similar_image is None):
                errors.append("No images found")
                                                    #print("The object (e.g., tattoo) exists in both images!")
                                               
        ##ofek061123

        # else:
        #     errors=errors+temp_err;
        return most_similar_image,int(facenum),max_similarity,errors;
    