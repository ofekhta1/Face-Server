import numpy as np
import os
from sklearn.metrics.pairwise import cosine_similarity
def get_all_detectors_faces(generated_embeddings:dict[str,np.ndarray],return_detector:str):
    detector_indices:dict[str,list[int]]={}
    for models in generated_embeddings:
        detector,embedder=models.split('_')
        if(detector!=return_detector):
            base_detector_embs = generated_embeddings[f"{return_detector}_{embedder}"]
            other_embeddings = generated_embeddings[f"{detector}_{embedder}"]
            # Swap the first and second rows
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


