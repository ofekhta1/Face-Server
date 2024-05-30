class BaseGenderAgeModel:
    def __init__(self):
        self.name="base_genderage"
        
    def get_gender_age(self,img,face):
        raise Exception("Get Gender Not Implemented")
