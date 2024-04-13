from .models.buffalo_l import Buffalo_L
from .models.base_model import BaseModel

class ModelLoader:
    models={"buffalo_l":Buffalo_L}

    instances={}

    @staticmethod
    def load_model(model_name)->BaseModel:
        if(model_name in ModelLoader.instances):
            return ModelLoader.instances[model_name]
        elif model_name in ModelLoader.models:
            model= ModelLoader.models[model_name]();
            ModelLoader.instances[model_name]=model
            return model
        #Model Doesnt exist!
        return None 

