import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
from insightface.utils.face_align import estimate_norm
from models.errors.base_error import BaseError

def get_all_detectors_faces(generated_embeddings:dict[str,np.ndarray],return_detector:str,model_loader)->dict[str,list]|BaseError:
    detector_indices:dict[str,list[int]]={}
  
    base_detector_embs=[]
    for models in generated_embeddings:
        detector,embedder=models.split('_')
        if f"{return_detector}_{embedder}" not in generated_embeddings:
            if return_detector not in model_loader.model_registry["detectors"]:
                raise Exception(f"The return_detector {return_detector} does not exist!")
            else:
                return BaseError(reason=f"No embeddings were created with return detector and the embedder {embedder}")
        if(detector!=return_detector):
            base_detector_embs = generated_embeddings[f"{return_detector}_{embedder}"]
            other_embeddings = generated_embeddings[f"{detector}_{embedder}"]
            if len(other_embeddings)>0:
                similarity_matrix = cosine_similarity(base_detector_embs, other_embeddings)
                print("Similarity Matrix:")
                print(similarity_matrix)
                detector_indices[detector]=convert_detector_indices(similarity_matrix);

    detector_indices[return_detector]=list(range(len(base_detector_embs)));

    return detector_indices;
def convert_detector_indices(similarity_matrix,tolerance=0.1):
    column_indexes = []
    for col_index, column in enumerate(similarity_matrix.T):  # Transpose the array to iterate over columns
        for i in range(len(column)):
            value=column[i]
            if abs(value - 1) < tolerance:
                column_indexes.append(i)
                break;
            if(i==len(column)-1):
                column_indexes.append(-1)
    replacement_value=max(column_indexes)+1
    for i in range(len(column_indexes)):
        if column_indexes[i]==-1:
            column_indexes[i]=replacement_value
            replacement_value+=1
    return column_indexes

def euclidean_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def are_bboxes_similar(bbox1, bbox2, threshold):
    return all(euclidean_distance(p1, p2) <= threshold for p1, p2 in zip(bbox1, bbox2))

def filter_faces(close_faces,far_faces):
    faces=far_faces.copy();
    for j in range(len(close_faces)):
        duplicate=False;
        for far_face in far_faces:
            if(are_bboxes_similar(close_faces[j]['bbox'],far_face['bbox'],20)):
                duplicate=True;
        if(not duplicate):
            faces.append(close_faces[j])
    return faces

#calculate the similarity between two images 
def calculate_similarity(emb_a, emb_b):
    
    similarity = np.dot(emb_a,emb_b) / (
        np.linalg.norm(emb_a) * np.linalg.norm(emb_b)
    )
    return similarity

def normalize_vector(v):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    normalized = v / norm
    return normalized

def string_to_numpy_array(string):
    # Removing brackets and splitting the string
    str_values = string.replace('[', '').replace(']', '').split()
    # Converting each string to a float
    float_values = [float(val) for val in str_values]
    # Converting the list of floats to a numpy array
    return np.array(float_values)

def norm_path(path):
    if os.name == 'nt':  # Windows
        return path.replace('/', '\\')
    else:  # Linux, macOS, etc.
        return path.replace('\\', '/')       

def transform_norm_landmarks(landmarks:np.ndarray):
    
    M=estimate_norm(lmk=landmarks,image_size=112);
    landmarks_homo = np.hstack([landmarks, np.ones((landmarks.shape[0], 1))])  # Convert to homogeneous coordinates
    transformed_landmarks:np.ndarray = np.dot(M, landmarks_homo.T).T
    return transformed_landmarks.tolist()

def calculate_quality(img:np.ndarray,face:dict)->float:
    bbox=face["bbox"]
    width=bbox[2]-bbox[0]
    height=bbox[3]-bbox[1]
    face_area=width*height
    img_area=img.shape[0]*img.shape[1]
    face_ratio=face_area/img_area
    return face_ratio;
