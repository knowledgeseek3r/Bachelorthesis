import os
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import plot_graph
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import torch


# loading config params
project_root =  "/mnt/md0/user/swidnickira68812/bertv1"
with open(project_root+"/config.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)


#-------------------------------       --------------------------------
#------------------------------- DATA  --------------------------------
#-------------------------------       --------------------------------
zusagen = params["data"]["zusagen"]
absagen = params["data"]["absagen"]

# Load data and set labels
data_zusagen = pd.read_csv(zusagen, error_bad_lines=False, encoding= 'unicode_escape')
data_zusagen['label'] = 0
data_absagen = pd.read_csv(absagen, error_bad_lines=False, encoding= 'unicode_escape')
data_absagen['label'] = 1





# Concatenate data
data = pd.concat([data_zusagen, data_absagen], axis=0).reset_index(drop=True)

# Splitt data for Hold-out cross validation
from sklearn.model_selection import train_test_split
X = data.lebenslauf.values
y = data.label.values

X_train, X_val, y_train, y_val =\
    train_test_split(X, y, test_size=params["data"]["test_size"], random_state=2020)



# Load test data
test_data = pd.read_csv('csv/con_testdata.csv', error_bad_lines=False, encoding= 'unicode_escape')

# Keep important columns
test_data = test_data[['lebenslauf']]


# Clean resume data by using re python library
def text_preprocessing(text):
    # Remove whitespace after question mark
    re.sub(r'([te\?])\s+', r'\1', text)

    # Remove question mark
    text = re.sub(r'\?', ' ', text).strip()

    # Remove all whitespaces between characters
    text= re.sub(r'[?:w\s?:w\s]*', '', text).strip()

    # Remove characters which appears more than once in a row
    text= re.sub(r'(\-)\1*', ' ', text).strip()

    # Remove all numbers
    # text= re.sub(r'[0-9]', ' ', text).strip()

    return text


#-------------------------------       --------------------------------
#------------------------------- TOKENIZER --------------------------------
#-------------------------------       --------------------------------

# Set model for tokenizing 
path_model = 'model/'

from transformers import BertTokenizer

# Load the BERT tokenizer
tokenizer = BertTokenizer.from_pretrained(path_model, local_files_only=True)

# Create a function to tokenize a set of texts
def preprocessing_for_bert(data):
    """Perform required preprocessing steps for pretrained BERT.
    @param    data (np.array): Array of texts to be processed.
    @return   input_ids (torch.Tensor): Tensor of token ids to be fed to a model.
    @return   attention_masks (torch.Tensor): Tensor of indices specifying which
                  tokens should be attended to by the model.
    """
    # Create empty lists to store outputs
    input_ids = []
    attention_masks = []

    # For every sentence...
    for sent in data:
        # `encode_plus` will:
        #    (1) Tokenize the sentence
        #    (2) Add the `[CLS]` and `[SEP]` token to the start and end
        #    (3) Truncate/Pad sentence to max length
        #    (4) Map tokens to their IDs
        #    (5) Create attention mask
        #    (6) Return a dictionary of outputs
        encoded_sent = tokenizer.encode_plus(
            text=text_preprocessing(sent),  # Preprocess sentence
            add_special_tokens=True,        # Add `[CLS]` and `[SEP]`
            max_length=MAX_LEN,             # Max length to truncate/pad
            truncation=True,                # Truncation on/off
            padding='max_length',           # Pad sentence to max length
            return_attention_mask=True      # Return attention mask
            )
        
        # Add the outputs to the lists
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    # Convert lists to tensors
    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


# Concatenate train data and test data
all_lebenslauf = np.concatenate([data.lebenslauf.values, test_data.lebenslauf.values])

# Encode our concatenated data
encoded_lebenslauf = [tokenizer.encode(sent, add_special_tokens=True) for sent in data.lebenslauf.values]

# Find the maximum length
max_len = max([len(sent) for sent in encoded_lebenslauf])
print('Max length: ', max_len)

# Specify `MAX_LEN`
MAX_LEN = params["data"]["max_length"]

lengths = []
for sent in encoded_lebenslauf:
    lengths.append(len(sent))

# Plot and save lengths of resume data
plot_graph.lengths(lengths)

# Run function `preprocessing_for_bert` on the train set and the validation set
print('Tokenizing data...')
train_inputs, train_masks = preprocessing_for_bert(X_train)
val_inputs, val_masks = preprocessing_for_bert(X_val)


#-------------------------------       --------------------------------
#------------------------------- DATALOADER --------------------------------
#-------------------------------       --------------------------------

# PyTorch DataLoader
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler

# Convert other data types to torch.Tensor
train_labels = torch.tensor(y_train)
val_labels = torch.tensor(y_val)

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

# Concatenate the train set and the validation set
full_train_data = torch.utils.data.ConcatDataset([train_data, val_data])
full_train_sampler = RandomSampler(full_train_data)
full_train_dataloader = DataLoader(full_train_data, sampler=full_train_sampler, batch_size=batch_size)