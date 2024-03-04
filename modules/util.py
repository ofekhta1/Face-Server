import numpy as np
from PIL import Image
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



