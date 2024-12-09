import json
from pydantic import BaseModel
class SimilarImage(BaseModel):
    image_name:str
    face_num:int
    similarity:float

    def __str__(self):
        return f"Image Name: {self.image_name}, Face Number: {self.face_num}, Similarity: {self.similarity}"
