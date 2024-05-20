class BaseGenderAgeModel:
    def __init__(self):
        self.name="base_genderage"
        
    def get_gender(self,img,face):
        raise Exception("Get Gender Not Implemented")
