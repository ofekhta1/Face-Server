import numpy as np
from insightface.utils.face_align import norm_crop
import os
from models.similar_image import SimilarImage
from . import util,image_embedding_manager,image_group_repository;
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import cv2
from  .model_loader import ModelLoader
from modules.models.base_model import BaseModel
import matplotlib.pyplot as plt


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
    # Load model on startup
    def __init__(self,groups:image_group_repository.ImageGroupRepository,emb_manager:image_embedding_manager.ImageEmbeddingManager, UPLOAD_FOLDER, STATIC_FOLDER):
        self.UPLOAD_FOLDER = UPLOAD_FOLDER
        self.STATIC_FOLDER = STATIC_FOLDER
        self.groups=groups;
        self.emb_manager=emb_manager;
    

    def __align_single_image(self, face, selected_face:int, filename:str, img:np.ndarray,model_name:str):
        landmarks = face["kps"].astype(int)
        aligned_filename = f"aligned_{selected_face}_{filename}"
        aligned_path = os.path.join(self.STATIC_FOLDER,model_name, aligned_filename)
        aligned_img = norm_crop(img, landmarks, 112, "arcface")
        cv2.imwrite(aligned_path, aligned_img)
        return aligned_filename



    def detect_faces_in_image(self, filename:str,model:BaseModel, images:list):
        img, faces = self.__extract_faces(filename,model)
        boxes=[]
        if faces:
            for face in faces:
                landmarks = face["kps"].astype(int)
                for point in landmarks:
                    cv2.circle(
                        img,
                        (int(point[0]), int(point[1])),
                        5,
                        (0, 0, 255),
                        -1,
                    )
                box=face['bbox'].astype(int).tolist();
                boxes.append(box)
            detected_filename = "detected_" + filename
            detected_path = os.path.join(self.STATIC_FOLDER,model.name, detected_filename)


            cv2.imwrite(detected_path, img)
            images.append(detected_filename)

           
        else:
            images.append(filename)
        return len(faces),boxes
        
   
    def create_aligned_images(self, filename:str,model:BaseModel, images:list,save=True)->tuple[np.typing.NDArray[np.uint8],list,list[str]]:
        img, faces = self.__extract_faces(filename,model)
        errors=[]
        if(not faces):
            print("No faces detected.")  # Debug log
            errors.append("No faces detected in one or both images.")
            return img,None,[],errors;

        for i in range(len(faces)):
            aligned_filename = self.__align_single_image(
                faces[i], i, filename, img,model.name
            )
            images.append(aligned_filename)
        return img,faces,errors;

    def __extract_faces(self, filename:str,model:BaseModel):
        if filename.startswith("aligned_") or filename.startswith("detected_"):
            path = os.path.join(self.STATIC_FOLDER,model.name, filename)
        else:
            path = os.path.join(self.UPLOAD_FOLDER, filename)
        img = cv2.imread(path)
        return img, model.extract_faces(img);


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
        
    def generate_all_emb(self,filename:str,model:BaseModel,save=True):
        errors=[];
        embeddings=[];
        if model:
            img,faces=self.__extract_faces(filename,model);                
            return self.generate_all_emb(img,faces,filename,model,save);
        else:
            errors.append("Error: Embedder model not initialized.")
        return embeddings,errors;

    def generate_all_emb(self,img,faces:list,filename:str,model:BaseModel,save=True)->tuple[np.typing.NDArray[np.uint8],list,list[np.ndarray],list[str]]:
        errors=[]
        if(not faces):
            print("No faces detected.")  # Debug log
            errors.append("No faces detected in one or both images.")
            return img,None,[],errors;
    
        aligned_images=[]
        embeddings=[]
        for i in range(len(faces)):
            embedding=model.embed(img,faces[i]);
            embeddings.append(np.array(embedding));
            aligned_filename = f"aligned_{i}_{filename}";

            aligned_images.append(aligned_filename)
            # face_count += 1


        #dedup code
        
        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix = np.clip(similarity_matrix, -1, 1)
        # Apply DBSCAN

        dbscan = DBSCAN(eps=0.1, min_samples=2, metric="precomputed")
        labels = dbscan.fit_predict(1 - similarity_matrix)  # Convert similarity to distance
        unique_values = np.unique(labels)[1:]#remove -1
        index_groups = {value: np.where(labels == value)[0] for value in unique_values}
        for group in index_groups:
            for dups_index in index_groups[group][1:]:
                embeddings.pop(dups_index)
                os.remove(os.path.join(self.STATIC_FOLDER,model.name,aligned_images[dups_index]))
                faces.pop(dups_index)
                aligned_images.pop(dups_index)
       
        if(save):
            for i in range(len(faces)):        
                self.emb_manager.add_embedding(embeddings[i],aligned_images[i],model.name);
        self.emb_manager.set_face_count(filename,len(faces),model_name=model.name)
        return img,faces,embeddings,errors;


    @staticmethod
    def points(numpoints,max_val,template_path,image_path):
      MIN_MATCH_COUNT = numpoints
      img1 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE) # queryImage
      img2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE) # trainImage
      sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=100)
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
          draw_params = dict(matchColor = (0,255,0), # draw matches in green color
      singlePointColor = None,
      matchesMask = matchesMask, # draw only inliers
      flags = 2)
          img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
        #   if 'blake' in image_path:
        #       plt.imshow(img3, 'gray')
        #       plt.show(block=True)
              
          #print("The object (e.g., tattoo) exists in both images!")
       else:
         #print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
         matchesMask = None
      
      draw_params = dict(matchColor = (0,255,0), # draw matches in green color
      singlePointColor = None,
      matchesMask = matchesMask, # draw only inliers
      flags = 2)
      img3 = cv2.drawMatches(img1,kp1,img2,kp2,good,None,**draw_params)
    #   plt.imshow(img3, 'gray')
    #   plt.show(block=True)
    #   if (len(good) >= MIN_MATCH_COUNT) and (max_val>= - 0.2 or len(good)>=60) and M is not None :
        #print("The object (e.g., tattoo) exists in both images!")
      
        #ImageHelper.create_combined_file()
    #   else:
    #  #plt.imshow(img3, 'gray')
    #  #plt.show(block=True)
    #      print("The object (e.g., tattoo) does NOT exist or the similarity is too low.")
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
    def allowed_file(filename):
        extension=os.path.splitext(filename)[1];
        return extension.lower() in ImageHelper.ALLOWED_EXTENSIONS;



    def generate_embedding(self,filename:str,selected_face:int,model:BaseModel,save=True)->np.ndarray[np.float32]:
        errors=[];
        embedding=None;
        model=ModelLoader.load_model("buffalo_l")
        if model:
            img,faces=self.__extract_faces(filename,model)
            if faces:
                if selected_face == -2 or len(faces) == 1:
                    i=0
                else:
                    i=selected_face
                embedding=model.embed(img,faces[i]);
                if(save):
                    self.emb_manager.add_embedding(embedding,f"aligned_{i}_{filename}",model.name);

            else:
                print("No faces detected.")  # Debug log
                errors.append("No faces detected in one or both images.")
        else:
            errors.append("Error: Embedder model not initialized.")
        return embedding,errors;

    def get_similar_images(self,user_embedding:list,filename:str,model_name:str,k=5):
        np_emb=np.array(user_embedding).astype("float32").reshape(1,-1)
        result=self.emb_manager.search(np_emb,k+1,model_name);
        filtered=[]
        seen_distances=[]
        for r in result:
            if r["distance"] not in seen_distances:
                seen_distances.append(r["distance"])
                i=r["index"];
                name=self.emb_manager.get_name(i,model_name);
                if(name.split('_')[-1]!=filename.split('_')[-1]):
                    filtered.append({"index":i,"name":name})
        valid=[x for x in filtered if len(emb := self.emb_manager.get_embedding(x['index'],model_name))>0 
                and not np.allclose(emb,user_embedding,rtol=1e-5,atol=1e-8)]
        return valid;
    def filter(self,threshold,model_name):
        manager=self.emb_manager
        errors=[]
        original_length=len(manager.db_embeddings[model_name].names);
        for name in manager.db_embeddings[model_name].names:
            embedding=manager.get_embedding_by_name(name,model_name)
            valid=self.get_similar_images(embedding,name.split('_')[-1],model_name);
            for image in valid:            
                match:str=image['name'];
                _,facenum,filename=match.split('_',2);
                similarity=util.calculate_similarity(
                    self.emb_manager.get_embedding(image['index'],model_name)
                    ,embedding);
                if(similarity>threshold):
                    manager.remove_embedding_by_index(image['index'],model_name);
        filtered_length=len(manager.db_embeddings[model_name].names);
        return original_length-filtered_length;

    def enhance_image(self,filename):
        image_path = os.path.join(self.UPLOAD_FOLDER, filename)
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            print(f"The image at {image_path} could not be loaded.")
            return

        # Apply slight Gaussian blur to the image to reduce noise
        blurred = cv2.GaussianBlur(image, (3, 3), 0)

        # Sharpen the image by subtracting the Gaussian blur from the original image
        sharpened = cv2.addWeighted(image, 1.5, blurred, -0.5, 0)

        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        lab = cv2.cvtColor(sharpened, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        l2 = clahe.apply(l)
        lab = cv2.merge((l2, a, b))
        enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        return enhanced
    
    def get_most_similar_image(self,selected_face:int,filename:str,model:BaseModel):
        user_image_path = os.path.join(self.UPLOAD_FOLDER, filename)
        errors=[]
        most_similar_image=None;
        most_similar_face=-2;
        max_similarity=-1;
        facenum=-2;
        temp_err=[]
        aligned_filename=f"aligned_{0 if selected_face == -2 else selected_face}_{filename}";
        embedding=self.emb_manager.get_embedding_by_name(aligned_filename,model.name)
        if len(embedding)>0:
            user_embedding=embedding;
        else:
            user_embedding,temp_err=self.generate_embedding(filename,selected_face,model);
            errors=errors+temp_err;
        if len(errors)==0:
            valid=self.get_similar_images(user_embedding,filename,model.name);
            for image in valid:
                try:  
                    match=image['name'];
                    _,facenum,filename=match.split('_',2);
                    similarity=util.calculate_similarity(
                        self.emb_manager.get_embedding(image['index'],model.name)
                        ,user_embedding);
                    if(similarity>max_similarity):
                        max_similarity=similarity;
                        most_similar_image=filename;
                        most_similar_face=int(facenum);
                except Exception as e:
                    # template_matching
                    print(f"failed to match image {match} because:\n{e}");
            if len(valid)==0:
                errors.append("No unique matching faces found!");

        errors=errors+temp_err;
        return most_similar_image,most_similar_face,max_similarity,errors;
    def get_k_similar_images(self,filename:str,selected_face:int,model:BaseModel,k=1)->tuple[list[SimilarImage],list[str]]:
        errors=[]
        similar_images=[];
        temp_err=[]
        aligned_filename=f"aligned_{0 if selected_face == -2 else selected_face}_{filename}";
        embedding=self.emb_manager.get_embedding_by_name(aligned_filename,model_name=model.name)
        if len(embedding)>0:
            user_embedding=embedding;
        else:
            user_embedding,temp_err=self.generate_embedding(filename,selected_face,model);
            errors=errors+temp_err;
        if len(errors)==0:
            valid=self.get_similar_images(user_embedding,filename=filename,model_name=model.name,k=k);
            for image in valid:
                try:  
                    match=image['name'];
                    _,facenum,filename=match.split('_',2);
                    similarity=util.calculate_similarity(
                        self.emb_manager.get_embedding(image['index'],model_name=model.name)
                        ,user_embedding);
                    similar_model= SimilarImage(filename,facenum,similarity)
                    similar_images.append(similar_model.to_json())
                except Exception as e:
                    # template_matching
                    print(f"failed to match image {match} because:\n{e}");
            if len(valid)==0:
                errors.append("No unique matching faces found!");

        errors=errors+temp_err;
        return similar_images,errors;

    def get_most_similar_image_by_template(self,filename):
        user_image_path = os.path.join(self.UPLOAD_FOLDER, filename)
        errors=[]
        most_similar_image=None;
        max_similarity=-1;
        box=[]
        template=cv2.imread(user_image_path)
        # grayTemplate=cv2.cvtColor(template,cv2.COLOR_BGR2GRAY)
        over20 = []
        max_len=0
        best_match_score = -1
        best_match_image_1 = None
        best_match_image_2=None 

        with os.scandir(self.UPLOAD_FOLDER) as entries:
            for entry in entries:
                if entry.is_file() and ImageHelper.allowed_file(entry.name):
                   # temp_template=grayTemplate.copy();
                    if (entry.name != filename) and (filename not in entry.name) and (entry.name != filename.replace("enhanced_", "")):
                        # if filename.endswith('.jpg') or filename.endswith('.png'):
                        #  image_path = os.path.join(image_dir, filename)
                        #  image = cv2.imread(image_path)
                        image = cv2.imread(entry.path)
                        if image is not None:
                            #  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                             img2 = image.copy()
                             scaling_factor = img2.shape[0] / template.shape[0]  
                             resized_template = cv2.resize(template, None, fx=scaling_factor, fy=scaling_factor, interpolation=cv2.INTER_AREA)
                             result = cv2.matchTemplate(img2, resized_template, cv2.TM_CCOEFF_NORMED)
                             min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                            #  if("GettyImages-1428204245_re_autoOrient_i" in entry.name):
                            #      i=1
                             if max_val > 0.3:
                              over20.append((max_val, entry.path))
 
                             if max_val > best_match_score and max_val <= 1:
                              best_match_score = max_val
                              best_match_image_1 = entry.name
                        
                        # img=cv2.imread(entry.path);
                        # if img is not None:
                        #  grayImage=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY);
                        #  if temp_template.shape[1] > img.shape[1] or temp_template.shape[0] >img.shape[0]:
                        #      temp_template = cv2.resize(temp_template, (img.shape[1],img.shape[0]))
                        #  result = cv2.matchTemplate(grayImage, temp_template, cv2.TM_CCOEFF_NORMED)
                        #  _, max_val, _ , max_loc = cv2.minMaxLoc(result)
                        #  full_path=os.path.join(self.UPLOAD_FOLDER,filename)
                        #  leng=  ImageHelper.points(4,max_similarity,full_path,entry.path)
                        #  #leng=  ImageHelper.points(4,max_similarity,full_path,entry.path)
                        #  if max_val >= max_similarity:
                        #      box=[max_loc[0],max_loc[1],template.shape[1]+max_loc[0],template.shape[0]+max_loc[1]]
                        #     # bottom_right = (top_left[0] + w, top_left[1] + h)
                        #     # cv2.rectangle(img, top_left, bottom_right, (0, 255, 0), 2)
                        #      most_similar_image = entry.name
                        #      max_similarity =max_val
                        #      #leng=  ImageHelper.points(4,max_similarity,full_path,entry.path)
            #cv2.destroyAllWindows();
        for file in over20:
         leng=  ImageHelper.points(4,file[0],user_image_path,file[1])
         if(max_len<leng):
           max_len=leng
           best_match_image_2=file[1]
           filenamesift = os.path.basename(best_match_image_2)
        if(best_match_image_1!=best_match_image_2):
         if(max_len>30):
           most_similar_image=filenamesift
         else:
          most_similar_image=best_match_image_1
        return most_similar_image,box,max_similarity,errors;
    
    def cluster_images(self,max_distance,min_samples,model_name)->dict[int,list[str]]:
        # Assuming 'embeddings' is a list of your 512-dimensional embeddings
        if(len(self.emb_manager.db_embeddings[model_name].embeddings)==0):
            return {}
        similarity_matrix = cosine_similarity(self.emb_manager.db_embeddings[model_name].embeddings)
        similarity_matrix = np.clip(similarity_matrix, -1, 1)
        # Apply DBSCAN

        dbscan = DBSCAN(eps=max_distance, min_samples=min_samples, metric="precomputed")
        labels = dbscan.fit_predict(1 - similarity_matrix)  # Convert similarity to distance
        unique_values = np.unique(labels)
        index_groups = {value: np.where(labels == value)[0] for value in unique_values}
        value_groups = {
            int(key): [self.emb_manager.db_embeddings[model_name].names[index] for index in indexes]
            for key, indexes in index_groups.items()
        }
        return value_groups;