import train
import model
import data_preparation
import plot_graph
import matplotlib.pyplot as plt
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler, dataloader
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import plot_roc_curve
from sklearn.metrics import accuracy_score, roc_curve, auc
from sklearn.model_selection import LeaveOneOut
import pandas as pd

# loading config params
project_root =  "/mnt/md0/user/swidnickira68812/bertv1"
with open(project_root+"/config.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

# Counter and lists for label values and prediction results
i = 0
all_y = []
all_probs=[]

# Load data and set labels
zusagen = params["data"]["zusagen"]
absagen = params["data"]["absagen"]

# Load data and set labels
data_zusagen = pd.read_csv(zusagen, error_bad_lines=False, encoding= 'unicode_escape')
data_zusagen['label'] = 0
data_absagen = pd.read_csv(absagen, error_bad_lines=False, encoding= 'unicode_escape')
data_absagen['label'] = 1

# Concatenate data
data = pd.concat([data_zusagen, data_absagen], axis=0).reset_index(drop=True)

X = data.lebenslauf.values
y = data.label.values

loo = LeaveOneOut()
loo.get_n_splits(X)

# Leave one out iteration
for train_index, test_index in loo.split(X):
    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]

    train_inputs, train_masks = data_preparation.preprocessing_for_bert(X_train)
    val_inputs, val_masks = data_preparation.preprocessing_for_bert(X_test)

    #-------------------------------       --------------------------------
    #------------------------------- DATALOADER --------------------------------
    #-------------------------------       --------------------------------

    # Convert other data types to torch.Tensor
    train_labels = torch.tensor(y_train)
    val_labels = torch.tensor(y_test)

    # For fine-tuning BERT, the authors recommend a batch size of 16 or 32.
    batch_size = params["training"]["batch_size"]

    # Create the DataLoader for our training set
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)

    # Create the DataLoader for our validation set
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)

    # Load Model for Leave one out evaluation 
    train.set_seed(params["training"]["seed"])    # Set seed for reproducibility
    model.bert_classifier, train.optimizer, train.scheduler = train.initialize_model(epochs=params["training"]["epochs"])
    train.train(model.bert_classifier, train_dataloader, val_dataloader, epochs=params["training"]["epochs"], evaluation=False)
    probs = train.bert_predict(model.bert_classifier, val_dataloader)
    
    # Add original y value and result for every prediction
    all_y.append(y_test)
    all_probs.append(probs)

    i = i +1
    print("\n Lebenslauf ",i, " wurde evaluiert... auf dem erstellten Modell.. fahre nun mit nächstem Durchlauf fort \n")

    # if i == 2:
    #     break;

array_y = np.concatenate( all_y, axis=0 )
array_probs = np.concatenate( all_probs, axis=0)
plot_graph.evaluate_roc(array_probs, array_y)

#----------------------------------- END OF Leave one out --------------------------------
