#from transformers.modeling_bert import BERT_PRETRAINED_MODEL_ARCHIVE_MAP
import train
import model
import data_preparation
import plot_graph
import matplotlib.pyplot as plt
import yaml
import numpy as np
import pandas as pd
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler



# loading config params
project_root =  "/mnt/md0/user/swidnickira68812/bertv1"
with open(project_root+"/config.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

batch_size = params["training"]["batch_size"]

# Method to read aresumes of the three categories and to predict the class of the resumes 
def splitted_prediction(cv_path):
    
    # declare list elements to save predition results
    prediction_school = []
    prediction_profession = []
    prediction_skills = []

    i = 0

    while i < 3:
        if i == 0:
            test_data= pd.read_csv(cv_path+'/test_absage_school.csv', error_bad_lines=False, encoding= 'unicode_escape')
            test_data = test_data[['lebenslauf']]
            prediction_school = create_dataloader_and_predict(test_data,"school")
            print("\n- - Prediction for school dataset done")
        if i == 1:
            test_data = pd.read_csv(cv_path+'/test_absage_prof.csv', error_bad_lines=False, encoding= 'unicode_escape')
            test_data = test_data[['lebenslauf']]
            prediction_profession = create_dataloader_and_predict(test_data,"profession")
            print("- - Prediction for profession dataset done")
        if i == 2:
            test_data = pd.read_csv(cv_path+'/test_absage_skills.csv', error_bad_lines=False, encoding= 'unicode_escape')
            test_data= test_data[['lebenslauf']] 
            prediction_skills = create_dataloader_and_predict(test_data,"skills")
            print("- - Prediction for skills dataset done\n")
        
        i = i +1

    return prediction_school, prediction_profession, prediction_skills

#NOTE: test_data parameter only needed for splitted_prediction function for bert v2 and v3
def create_dataloader_and_predict(test_data, modelname,varianttyp="bertv2"):
        ### Predictions on Test Set
        train.set_seed(params["training"]["seed"])    # Set seed for reproducibility

        if varianttyp=="bertv1":
            test_data = pd.read_csv('csv/test_gesehen.csv', error_bad_lines=False, encoding= 'unicode_escape')
            bert_classifier, optimizer, scheduler = train.initialize_model(modelname="default")
            test_data= test_data[['lebenslauf']] 

        #Tokenizing data
        test_inputs, test_masks = data_preparation.preprocessing_for_bert(test_data.lebenslauf)

        # Create the DataLoader for our test set
        test_dataset = TensorDataset(test_inputs, test_masks)
        test_sampler = SequentialSampler(test_dataset)
        test_dataloader = DataLoader(test_dataset, sampler=test_sampler, batch_size=batch_size)

        # Compute predicted probabilities on the test set
        if varianttyp!="bertv1":
            bert_classifier, optimizer, scheduler = train.initialize_model(modelname=modelname)
        
        probs = train.bert_predict(bert_classifier, test_dataloader)

        return probs

# Test for Bert variant 1
plot_graph.single_score(create_dataloader_and_predict("","default","bertv1"))   


# # Test for Bert variant 2
# prediction_school, prediction_profession, prediction_skills = splitted_prediction("csv")
# plot_graph.final_score(prediction_school, prediction_profession, prediction_skills, "heuristic")
