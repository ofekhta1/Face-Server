import numpy as np
from insightface.utils.face_align import norm_crop
import os

from .stores import in_memory_image_embedding_manager,image_group_repository
from modules.models import ModelLoader
from insightface.app.common import Face
# 
from models.stored_embedding import FaceEmbedding
from models.similar_image import SimilarImage
from . import util;
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity
import cv2
import time
from modules.models import BaseGenderAgeModel,BaseDetectorModel,BaseEmbedderModel,FamilyClassifier


class ImageHelper:

    ALLOWED_EXTENSIONS = {
        ".png",
        ".jpg",
        ".jpeg",
        ".gif",
        ".bmp",
        ".tif",
        ".tiff",
        ".webp",
    }

    # Load model on startup
    def __init__(
        self,
        groups: image_group_repository.ImageGroupRepository,
        emb_manager: in_memory_image_embedding_manager.InMemoryImageEmbeddingManager,
        UPLOAD_FOLDER,
        STATIC_FOLDER,
    ):
        self.UPLOAD_FOLDER = UPLOAD_FOLDER
        self.STATIC_FOLDER = STATIC_FOLDER
        self.groups = groups
        self.emb_manager = emb_manager

    def __align_single_image(
        self,
        face,
        selected_face: int,
        filename: str,
        img: np.ndarray,
        detector_name: str,
    ):
        landmarks = face["kps"].astype(int)
        aligned_filename = f"aligned_{selected_face}_{filename}"
        aligned_path = os.path.join(self.STATIC_FOLDER, detector_name, aligned_filename)
        aligned_img = norm_crop(img, landmarks, 112, "arcface")
        cv2.imwrite(aligned_path, aligned_img)
        return aligned_filename

    def get_face_boxes(self, filename: str, detector_name: str,embedder_name:str) -> list[list[int]]:
        boxes = self.emb_manager.get_image_boxes(filename, detector_name,embedder_name=embedder_name)
        return boxes

    def detect_faces_in_image(
        self, filename: str, model: BaseDetectorModel, images: list
    ):
        img, faces = self.__extract_faces(filename, model)
        boxes = []
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
                box = face["bbox"].astype(int).tolist()
                boxes.append(box)
            detected_filename = "detected_" + filename
            detected_path = os.path.join(
                self.STATIC_FOLDER, model.name, detected_filename
            )

            cv2.imwrite(detected_path, img)
            images.append(detected_filename)

        else:
            images.append(filename)
        return len(faces), boxes

    def create_aligned_images(
        self, filename: str, detector: BaseDetectorModel, images: list
    ) -> tuple[np.typing.NDArray[np.uint8], list, list[str]]:
        img, faces = self.__extract_faces(filename, detector)
        errors = []
        if not faces:
            print("No faces detected.")  # Debug log
            errors.append("No faces detected in one or both images.")
            return img, None, errors

        for i in range(len(faces)):
            aligned_filename = self.__align_single_image(
                faces[i], i, filename, img, detector.name
            )
            images.append(aligned_filename)
        return img, faces, errors

    def __load_image(self, filename: str, model: BaseDetectorModel):
        if filename.startswith("aligned_") or filename.startswith("detected_"):
            path = os.path.join(self.STATIC_FOLDER, model.name, filename)
        else:
            path = os.path.join(self.UPLOAD_FOLDER, filename)
        img = cv2.imread(path)
        return img       
    def __extract_faces(self, filename: str, model: BaseDetectorModel):
        img =self.__load_image(filename,model)
        return img, model.extract_faces(img)
       

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

    def generate_all_emb(
        self,
        filename: str,
        detector: BaseDetectorModel,
        embedder: BaseEmbedderModel,
        save=True,
    ):
        errors = []
        embeddings = []
        if detector:
            img, faces = self.__extract_faces(filename, detector)
            return self.generate_all_emb(img, faces, filename, detector, embedder, save)
        else:
            errors.append("Error: detector model not initialized.")
        return embeddings, errors
   
    def generate_all_emb(
        self,
        img,
        faces: list,
        filename: str,
        detector: BaseDetectorModel,
        embedder: BaseEmbedderModel,
        gender_age:BaseGenderAgeModel=None
    ) -> tuple[np.typing.NDArray[np.uint8], list[FaceEmbedding], list[str]]:
        errors = []
        if not faces:
            print("No faces detected.")  # Debug log
            errors.append("No faces detected in one or both images.")
            return img, None, errors

        aligned_images = []
        embeddings = []

        for i in range(len(faces)):
            embedding = embedder.embed(img, faces[i])
            embedding=np.array(embedding)
            norm = np.linalg.norm(embedding)
            embedding=embedding/norm;
            embeddings.append(embedding)
            aligned_filename = f"aligned_{i}_{filename}"

            aligned_images.append(aligned_filename)
            # face_count += 1

        # internal dedup code

        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix = np.clip(similarity_matrix, -1, 1)
        # Apply DBSCAN

        dbscan = DBSCAN(eps=0.1, min_samples=2, metric="precomputed")
        labels = dbscan.fit_predict(
            1 - similarity_matrix
        )  # Convert similarity to distance
        clusters = np.unique(labels)
        clusters = np.delete(
            clusters, np.where(clusters == -1)
        )  # remove -1, non cluster data
        index_groups = {value: np.where(labels == value)[0] for value in clusters}
        for group in index_groups:
            # remove all values except for first value per cluster(dups)
            for dups_index in index_groups[group][1:]:
                os.remove(
                    os.path.join(
                        self.STATIC_FOLDER, detector.name, aligned_images[dups_index]
                    )
                )
                embeddings[dups_index] = None
                faces[dups_index] = None
                aligned_images[dups_index] = None

        filtered_embeddings = [e for e in embeddings if e is not None]
        filtered_faces = [f for f in faces if f is not None]
        filtered_aligned_images = [ai for ai in aligned_images if ai is not None]
        face_embeddings=[]
        for i in range(len(filtered_faces)):
            f=FaceEmbedding(filtered_aligned_images[i],[int(coord) for coord in filtered_faces[i]['bbox']],filtered_embeddings[i])
            if(gender_age):
                f.gender,f.age=gender_age.get_gender_age(img,filtered_faces[i]);          
            face_embeddings.append(f);
        #check if embeddings exist in emb_manager,in batch
        query=np.array([f.embedding for f in face_embeddings])
        similar=self.emb_manager.search(query,1,detector.name,embedder.name)
        distance_dup_thresh=0.95
        for i in range(len(similar)):
            closest=similar[i][0]
            if closest['distance']> distance_dup_thresh:
                face_embeddings[i].is_dup=True

        
        # self.emb_manager.set_face_count(filename,len(filtered_faces),detector_name=model.name)
        return img, face_embeddings, errors

    @staticmethod
    def points(numpoints, max_val, template_path, image_path):
        MIN_MATCH_COUNT = numpoints
        img1 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)  # queryImage
        img2 = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # trainImage
        sift = cv2.SIFT_create(contrastThreshold=0.02, edgeThreshold=100)
        kp1, des1 = sift.detectAndCompute(img1, None)
        kp2, des2 = sift.detectAndCompute(img2, None)
        FLANN_INDEX_KDTREE = 1
        index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
        search_params = dict(checks=50)
        flann = cv2.FlannBasedMatcher(index_params, search_params)
        matches = flann.knnMatch(des1, des2, k=2)
        good = []
        M = 0
        for m, n in matches:
            if m.distance < 0.7 * n.distance:
                good.append(m)
            if (len(good) > MIN_MATCH_COUNT) and (max_val >= -0.2 or len(good) >= 60):
                src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(
                    -1, 1, 2
                )
                dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(
                    -1, 1, 2
                )
                M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
                matchesMask = mask.ravel().tolist()
                h, w = img1.shape
                pts = np.float32(
                    [[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]
                ).reshape(-1, 1, 2)
                if M is not None:
                    dst = cv2.perspectiveTransform(pts, M)
                    img2 = cv2.polylines(
                        img2, [np.int32(dst)], True, 255, 3, cv2.LINE_AA
                    )
                    draw_params = dict(
                        matchColor=(0, 255, 0),  # draw matches in green color
                        singlePointColor=None,
                        matchesMask=matchesMask,  # draw only inliers
                        flags=2,
                    )
                    img3 = cv2.drawMatches(
                        img1, kp1, img2, kp2, good, None, **draw_params
                    )
            #   if 'blake' in image_path:
            #       plt.imshow(img3, 'gray')
            #       plt.show(block=True)

            # print("The object (e.g., tattoo) exists in both images!")
            else:
                # print( "Not enough matches are found - {}/{}".format(len(good), MIN_MATCH_COUNT) )
                matchesMask = None

        draw_params = dict(
            matchColor=(0, 255, 0),  # draw matches in green color
            singlePointColor=None,
            matchesMask=matchesMask,  # draw only inliers
            flags=2,
        )
        img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, **draw_params)
        #   plt.imshow(img3, 'gray')
        #   plt.show(block=True)
        #   if (len(good) >= MIN_MATCH_COUNT) and (max_val>= - 0.2 or len(good)>=60) and M is not None :
        # print("The object (e.g., tattoo) exists in both images!")

        # ImageHelper.create_combined_file()
        #   else:
        #  #plt.imshow(img3, 'gray')
        #  #plt.show(block=True)
        #      print("The object (e.g., tattoo) does NOT exist or the similarity is too low.")
        if M is not None:
            return len(good)
        else:
            return 0

    def create_combined_file(max_loc, image, template):
        h, w = image.shape[:2]
        top_left = max_loc
        bottom_right = (top_left[0] + w, top_left[1] + h)
        detected_object = template[
            top_left[1] : bottom_right[1], top_left[0] : bottom_right[0]
        ]
        detected_object_resized = cv2.resize(
            detected_object, (image.shape[1], image.shape[0])
        )
        combined_image = np.hstack((image, detected_object_resized))

    # output_dir = create_output_directory(image_files, images_dir + r"\archive")
    # combined_image_path = os.path.join(output_dir, "combined_" + image_files)
    # cv2.imwrite(combined_image_path, combined_image)

    @staticmethod
    def allowed_file(filename: str):
        extension = os.path.splitext(filename)[1]
        return extension.lower() in ImageHelper.ALLOWED_EXTENSIONS

    def generate_embedding(
        self,
        filename: str,
        selected_face: int,
        detector: BaseDetectorModel,
        embedder: BaseEmbedderModel,
        save=True,
    ) -> np.ndarray[np.float32]:
        errors = []
        embedding = None
        if embedder and detector:
            img, faces = self.__extract_faces(filename, detector)
            if faces:
                if selected_face == -2 or len(faces) == 1:
                    i = 0
                else:
                    i = selected_face
                embedding = embedder.embed(img, faces[i])
                if save:
                    box = faces[i]["bbox"].astype(int).tolist()
                    self.emb_manager.add_embedding(
                        embedding,
                        f"aligned_{i}_{filename}",
                        box,
                        detector.name,
                        embedder.name,
                    )

            else:
                print("No faces detected.")  # Debug log
                errors.append("No faces detected in one or both images.")
        else:
            errors.append("Error: Embedder model not initialized.")
        return embedding, errors

    def get_similar_images(
        self,
        user_embedding: list,
        filename: str,
        detector_name: str,
        embedder_name: str,
        k=5,
    ):
        np_emb = np.array(user_embedding).astype("float32").reshape(1, -1)

        start = time.time()

        result = self.emb_manager.search(np_emb, k + 1, detector_name, embedder_name)
        end = time.time()
        print(f"Elapsed Search Time: {(end - start)*1000} ms")
        filtered = []
        seen_distances = []
        for r in result[0]:
            if r["distance"] not in seen_distances:
                seen_distances.append(r["distance"])
                i = r["index"]
                name = r['Embedding'].name
                if name.split("_")[-1] != filename.split("_")[-1]:
                    filtered.append({"index": i, "name": name,"Embedding":r['Embedding']})
        valid = [
            x
            for x in filtered
            if len(
                emb := x["Embedding"].embedding
            )
            > 0
            and not np.allclose(emb, np_emb, rtol=1e-5, atol=1e-8)
        ]
        return valid

    def filter(self, threshold, detector_name, embedder_name):
        manager = self.emb_manager
        errors = []
        embeddings = (
            manager.get_all_embeddings(detector_name,embedder_name)
        )
        original_length = len(embeddings)
        for embedding in embeddings:
            valid = self.get_similar_images(
                embedding.embedding,
                embedding.name.split("_")[-1],
                detector_name,
                embedder_name,
            )
            for image in valid:
                match: str = image["name"]
                _, facenum, filename = match.split("_", 2)
                similarity = util.calculate_similarity(
                    self.emb_manager.get_embedding(
                        image["index"], detector_name, embedder_name
                    ).embedding,
                    embedding.embedding,
                )
                if similarity > threshold:
                    manager.remove_embedding_by_index(
                        image["index"], detector_name, embedder_name
                    )
        filtered_length = len(embeddings)
        return original_length - filtered_length

    def enhance_image(self, filename):
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

    def get_k_similar_images(
        self,
        filename: str,
        selected_face: int,
        threshold: float,
        detector: BaseDetectorModel,
        embedder: BaseEmbedderModel,
        k=1,
    ) -> tuple[list[SimilarImage], list[str]]:
        errors = []
        similar_images = []
        temp_err = []
        aligned_filename = (
            f"aligned_{0 if selected_face == -2 else selected_face}_{filename}"
        )
        start = time.time()
        embedding = self.emb_manager.get_embedding_by_name(
            aligned_filename, detector_name=detector.name, embedder_name=embedder.name
        )
        end = time.time()
        print(f"Elapsed Get Embedding Time: {(end - start)*1000}ms")
        if len(embedding.embedding) > 0:
            user_embedding = embedding.embedding
        else:
            user_embedding, temp_err = self.generate_embedding(
                filename, selected_face, detector, embedder
            )
            errors = errors + temp_err
        if len(errors) == 0:
            start = time.time()

        
            valid = self.get_similar_images(
                user_embedding,
                filename=filename,
                detector_name=detector.name,
                embedder_name=embedder.name,
                k=k,
            )
            end = time.time()
            print(f"Elapsed Similar Images Time: {(end - start)*1000}ms")
            for image in valid:
                try:
                    match = image["name"]
                    _, facenum, filename = match.split("_", 2)
                    similarity = util.calculate_similarity(
                        image['Embedding'].embedding,
                        user_embedding,
                    )
                    if similarity > threshold:
                        similar_model = SimilarImage(filename, int(facenum), similarity)
                        similar_images.append(similar_model)
                except Exception as e:
                    # template_matching
                    print(f"failed to match image {match} because:\n{e}")
            if len(valid) == 0:
                errors.append("No unique matching faces found!")
            elif len(similar_images) == 0:
                errors.append(f"No matching faces found with sufficient similarity")

        errors = errors + temp_err
        return similar_images, errors

    #function that returns the most similiar image to the selected template, accorfing to the selected thr and sift points
    def get_most_similar_image_by_template(self, filename,similarity_thresh):
        user_image_path = os.path.join(self.UPLOAD_FOLDER, filename)
        errors = []
        most_similar_image = None
        box = []
        template = cv2.imread(user_image_path, cv2.IMREAD_GRAYSCALE)
        over20 = []
        max_len = 0
        best_match_score = -1
        best_match_score2= -1
        best_score_sofi= -1
        best_match_image_1 = None
        best_match_image_2 = None

        with os.scandir(self.UPLOAD_FOLDER) as entries:
            for entry in entries:
                if entry.is_file() and ImageHelper.allowed_file(entry.name):
                    if (
                        (entry.name != filename)
                        and (filename not in entry.name)
                        and (entry.name != filename.replace("enhanced_", ""))
                    ):
                        image = cv2.imread(entry.path, cv2.IMREAD_GRAYSCALE)
                        if image is not None:
                            img2 = image.copy()
                            scaling_factor = img2.shape[0] / template.shape[0]
                            resized_template = cv2.resize(
                                template,
                                None,
                                fx=scaling_factor,
                                fy=scaling_factor,
                                interpolation=cv2.INTER_AREA,
                            )
                            result = cv2.matchTemplate(
                                img2, resized_template, cv2.TM_CCOEFF_NORMED
                            )
                            min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
                            if max_val > 0.3:
                                over20.append((max_val, entry.path))

                            if max_val > best_match_score and max_val <= 1:
                                best_match_score = max_val
                                best_match_image_1 = entry.name
                                box = [
                                    max_loc[0],
                                    max_loc[1],
                                    template.shape[1] + max_loc[0],
                                    template.shape[0] + max_loc[1],
                                ]

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
            # cv2.destroyAllWindows();
       #run over all the files with the appropiate similarity( accorfing to the configured threshold)
        for file in over20:
            leng = ImageHelper.points(4, file[0], user_image_path, file[1])
            if max_len < leng:
                max_len = leng
                best_match_image_2 = file[1]
                filenamesift = os.path.basename(best_match_image_2)
        if best_match_image_1 != best_match_image_2:
            if max_len > 30:
                most_similar_image = filenamesift
            else:
                most_similar_image = best_match_image_1
        return most_similar_image, box, best_match_score, errors
        

    def cluster_images(
        self, max_distance, min_samples, detector_name, embedder_name
    ) -> dict[int, list[str]]:
        # Assuming 'embeddings' is a list of your 512-dimensional embeddings
        face_embeddings =self.emb_manager.get_all_embeddings(detector_name,embedder_name)
        embeddings=[e.embedding for e in face_embeddings]
        if len(embeddings) == 0:
            return {}
        #create a similarity matrix that includes the cosine similarity between every 2 images in the embedding data
        similarity_matrix = cosine_similarity(embeddings)
        similarity_matrix = np.clip(similarity_matrix, -1, 1)
        # Apply DBSCAN-model that takes the min sample and max distance as returns the groups according to the required distances and min images in group parameters

        dbscan = DBSCAN(eps=max_distance, min_samples=min_samples, metric="precomputed")
        labels = dbscan.fit_predict(1 - similarity_matrix)  # Convert similarity to distance
        unique_values = np.unique(labels)
        index_groups = {value: np.where(labels == value)[0] for value in unique_values}
        value_groups = {
            int(key): [
                face_embeddings[index].name
                for index in indices
            ]
            for key, indices in index_groups.items()
        }
        return value_groups;
    def cluster_images_family(self, max_distance, min_samples, detector_name:str, embedder_name:str,classifier:FamilyClassifier) -> dict[int, list[str]]:
        embeddings=[]
        embeddings=self.emb_manager.get_all_embeddings(detector_name=detector_name,embedder_name=embedder_name);
        Genders = []
        if len(embeddings) == 0:
            return {}
        similarity_matrix = cosine_similarity([e.embedding for e in embeddings])
        Genders=[e.gender for e in embeddings]
        is_same_family = classifier.predict_batch(similarity_matrix, Genders)
        distance_matrix = 1 - is_same_family
        dbscan = DBSCAN(eps=max_distance, min_samples=min_samples, metric='precomputed')
        labels = dbscan.fit_predict(distance_matrix)
        value_groups = {}
        for label in np.unique(labels):
        #  if label != -1:  # Exclude noise points
            value_groups[int(label)] = [embeddings[i].name for i in range(len(labels)) if labels[i] == label]

        return value_groups

    
            


       
