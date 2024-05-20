import pickle
import os;
import pandas as pd;
import numpy as np;
class FamilyClassifier:
    # load the weights of the family_classifier_nodel
    def __init__(self,root_path):
        models_path=os.path.join(root_path,"OnnxModels");
        with open(os.path.join(models_path,"family_classifier_model.pkl"), "rb") as model_file:
            self.model = pickle.load(model_file);
        with open(os.path.join(models_path,"scaler.pkl"), "rb") as scaler_file:
            self.scaler=pickle.load(scaler_file);

    def predict(self,similarity,gender1,gender2):
        new_data = {
        "Gender1": gender1,
        "Gender2": gender2,
        "similatrity": similarity
        # "landmarks1": landmarks1,
        # "landmarks2": landmarks2
        }
        features =self.__create_features(pd.DataFrame([new_data]))
        features_scaled = self.scaler.transform(features)
        prediction = self.model.predict(features_scaled)
        return prediction[0],similarity
        
    def __create_features(self,df):
        features = []
        for _, row in df.iterrows():
            gender_pic1_numeric =self.__map_gender(row['Gender1'])
            gender_pic2_numeric =self.__map_gender(row['Gender2'])
            similarity_score = row['similatrity']  # Make sure this column name is correct
            # landmarks1 = row['landmarks1'].flatten()  # Flatten the landmarks array
            # landmarks2 = row['landmarks2'].flatten()  # Flatten the landmarks array

            # Combine the features into a single feature array
            feature = np.concatenate(([gender_pic1_numeric, gender_pic2_numeric, similarity_score],))
            features.append(feature)

        return np.array(features)
    #  invert male and female
    def __map_gender(self,gender):
        if(gender==1):
            return 0
        elif(gender==0):
            return 1
        else:
            return -1
