import json
class SimilarImage:
    def __init__(self, image_name:str, face_num:int, similarity:float):
        self.image_name = image_name
        self.face_num = face_num
        self.similarity = similarity
    def __str__(self):
        return f"Image Name: {self.image_name}, Face Number: {self.face_num}, Similarity: {self.similarity}"

    def to_json(self):
        return {
            "image_name": self.image_name,
            "face_num": self.face_num,
            "similarity": str(self.similarity)
        }

    @classmethod
    def from_json(cls, json_data):
        return cls(
            json_data["image_name"],
            json_data["face_num"],
            json_data["similarity"]
        )
