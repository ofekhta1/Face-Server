class BaseGenderAgeModel:
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    def __init__(self):
        self.name="base_genderage"
        
    def get_gender_age(self,img,face):
        raise Exception("Get Gender Not Implemented")

    def get_gender_age_raw(self,img):
        raise Exception("Get GenderAge raw Not Implemented")
