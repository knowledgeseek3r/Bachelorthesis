import os
from typing import final
import re
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yaml
import plot_graph
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import data_preparation
import early_stopping_accuracy
import early_stopping
import model
import torch
import torch.nn as nn

# loading config params
project_root =  "/mnt/md0/user/swidnickira68812/bertv1"
with open(project_root+"/config.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

#Turn on/off saving model option based on highest accuracy each epoch or lowest validation loss
save_highest_accuracy = bool(params["training"]["save_highest_accuracy"])
save_lowest_loss = bool(params["training"]["save_lowest_loss"])

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

#Set GPU as device for training
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print("There are" , {torch.cuda.device_count()},  "GPU(s) available.")
    print("Device name:", torch.cuda.get_device_name(0))

else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")

#-------------------------------       --------------------------------
#------------------------------- OPTIMIZER --------------------------------
#-------------------------------       --------------------------------

# Optional enabling of expoential learning rate 
exp_lr = params["training"]["exp_lr"]

from transformers import AdamW, get_linear_schedule_with_warmup

def initialize_model(epochs=30,modelname="default"):
    """Initialize the Bert Classifier, the optimizer and the learning rate scheduler.
    """
    # Instantiate Bert Classifier
    bert_classifier = model.BertClassifier(freeze_bert=params["model"]["freezelayers"], modelname="default")

    # Tell PyTorch to run the model on GPU
    bert_classifier.to(device)

    # Create the optimizer
    optimizer = AdamW(bert_classifier.parameters(),
                      lr=float(params["model"]["learning_rate"]),
                      eps=1e-8,
                      weight_decay=float(params["model"]["weight_decay"])  
                      )

    # Total number of training steps
    total_steps = len(data_preparation.train_dataloader) * epochs

    # Choose between usage or not usage of exponential rate 
    if exp_lr == False:
        #Set up the learning rate scheduler
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=0, # Default value
                                                    num_training_steps=total_steps)
    else:
        decayRate = 0.9
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer=optimizer, gamma=decayRate)
        print("\nExponential Learning Rate ACTIVATED\n")

    return bert_classifier, optimizer, scheduler

#-------------------------------       --------------------------------
#------------------------------- TRAIN --------------------------------
#-------------------------------       --------------------------------

import random
import time

def set_seed(seed_value=100):
    """Set seed for reproducibility.
    """
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    torch.cuda.manual_seed_all(seed_value)

# Specify loss function
loss_fn = nn.CrossEntropyLoss()

#Early stopping instance
#Validation accuracy

early_stopping_accuracy = early_stopping_accuracy.EarlyStopping()

#Validation loss
early_stopping= early_stopping.EarlyStopping()


#Variables for plotting epoch/loss
p_epoch = [None] * 0
p_valloss = [None] * 0
p_trainloss = [None] * 0
p_valaccuracy = [None] * 0
p_trainaccuracy = [None] * 0

# Method for training
def train(model, train_dataloader, val_dataloader=None, epochs=30, evaluation=False):
    """Train the BertClassifier model.
    """
    # Start training loop
    print("Start training...\n")
    for epoch_i in range(epochs):
        # =======================================
        #               Training
        # =======================================
        # Print the header of the result table
        print(f"{'Epoch':^7} | {'Batch':^7} | {'Train Loss':^12} | {'Val Loss':^10} | {'Train Acc':^9} | {'Val Acc':^9} | {'Elapsed':^9}")
        print("-"*82)

        # Measure the elapsed time of each epoch
        t0_epoch, t0_batch = time.time(), time.time()

        # Reset tracking variables at the beginning of each epoch
        total_loss, batch_loss, batch_counts = 0, 0, 0
        train_accuracy = []

        # Put the model into the training mode
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            # Load batch to GPU
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

            # Zero out any previously calculated gradients
            model.zero_grad()

            # Perform a forward pass. This will return logits.
            logits = model(b_input_ids, b_attn_mask)

            # Compute loss and accumulate the loss values
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()

            # Perform a backward pass to calculate gradients
            loss.backward()

            # Clip the norm of the gradients to 1.0 to prevent "exploding gradients"
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and the learning rate
            optimizer.step()
            scheduler.step()

            # Get the predictions
            preds = torch.argmax(logits, dim=1).flatten()

            # Calculate the accuracy rate
            accuracy = (preds == b_labels).cpu().numpy().mean() * 100
            train_accuracy.append(accuracy)
        

            # Print the loss values and time elapsed for every 20 batches
            if (step % 20 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                # Calculate time elapsed for 20 batches
                time_elapsed = time.time() - t0_batch

                # Print training results
                print(f"{epoch_i + 1:^7} | {step:^7} | {batch_loss / batch_counts:^12.6f} | {'-':^10} | {'-':^9} | {'-':^9} | {time_elapsed:^9.2f}")

                # Reset batch tracking variables
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()

        # Calculate the average loss over the entire training data
        avg_train_loss = total_loss / len(train_dataloader)
        # Compute the average accuracy and loss over the validation set.
        p_trainaccuracy.append((np.mean(train_accuracy)) / 100)
        print("-"*82)
        # =======================================
        #               Evaluation
        # =======================================
        if evaluation == True:
            # After the completion of each training epoch, measure the model's performance
            # on our validation set.
            val_loss, val_accuracy = evaluate(model, val_dataloader)

            #Early stopping --> saving best model

            if save_highest_accuracy:
                #Validation accuracy
                early_stopping_accuracy(val_accuracy, model)

            if save_lowest_loss:
                #Validation loss
                early_stopping(val_accuracy, model)
        

            # Print performance over the entire training data
            time_elapsed = time.time() - t0_epoch
            
            print(f"{epoch_i + 1:^7} | {'-':^7} | {avg_train_loss:^12.6f} | {val_loss:^10.6f} | {(np.mean(train_accuracy) / 100):^9.2f} | {val_accuracy:^9.2f} | {time_elapsed:^9.2f} ")
            print("-"*82)
           
            #Needed for plotting epoch/loss graph
            p_trainloss.append(avg_train_loss)
            p_epoch.append(epoch_i)
            p_valloss.append(val_loss)
            
            #Needed for plotting epoch/accuracy graph
            p_valaccuracy.append(val_accuracy/100)

            

        print("\n")
    

    print("Training complete!")
    print("Train Acc: " + str(p_trainaccuracy[-1]))


#-------------------------------       --------------------------------
#------------------------------- EVALUATION --------------------------------
#-------------------------------       --------------------------------


# Method for evaluation
def evaluate(model, val_dataloader):
    """After the completion of each training epoch, measure the model's performance
    on our validation set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    # Tracking variables
    val_accuracy = []
    val_loss = []

    # For each batch in our validation set...
    for batch in val_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)

        # Compute loss
        loss = loss_fn(logits, b_labels)
        val_loss.append(loss.item())

        # Get the predictions
        preds = torch.argmax(logits, dim=1).flatten()

        # Calculate the accuracy rate
        accuracy = (preds == b_labels).cpu().numpy().mean() * 100
        val_accuracy.append(accuracy)

    # Compute the average accuracy and loss over the validation set.
    val_loss = np.mean(val_loss)
    val_accuracy = np.mean(val_accuracy)

    return val_loss, val_accuracy


#-------------------------------       --------------------------------
#------------------------------- PREDICTION --------------------------------
#-------------------------------       --------------------------------


# Evaluation Set
import torch.nn.functional as F

# Predict label and output softmax
def bert_predict(model, test_dataloader):
    """Perform a forward pass on the trained BERT model to predict probabilities
    on the test set.
    """
    # Put the model into the evaluation mode. The dropout layers are disabled during
    # the test time.
    model.eval()

    all_logits = []

    # For each batch in our test set...
    for batch in test_dataloader:
        # Load batch to GPU
        b_input_ids, b_attn_mask = tuple(t.to(device) for t in batch)[:2]

        # Compute logits
        with torch.no_grad():
            logits = model(b_input_ids, b_attn_mask)
        all_logits.append(logits)
    
    # Concatenate logits from each batch
    all_logits = torch.cat(all_logits, dim=0)

    # Apply softmax to calculate probabilities
    probs = F.softmax(all_logits, dim=1).cpu().numpy()

    # Sigmoid 
    s_probs = torch.sigmoid(all_logits)

    return probs