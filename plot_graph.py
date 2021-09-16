import itertools
from typing import final
import numpy as np
import matplotlib.pyplot as plt

from sklearn import svm, datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

# Print all measures of the evualation
def bert_measures(cnf_matrix):
    TN = cnf_matrix[0][0]
    FN = cnf_matrix[1][0]
    TP = cnf_matrix[1][1]
    FP = cnf_matrix[0][1]


# Function to print train and loss
def loss(p_trainloss, p_epoch, p_valloss):
    plt.title('Loss and Epoch')
    plt.plot(p_epoch,p_valloss, color='orange', label="val loss")
    plt.plot(p_epoch,p_trainloss,"-b", label="train loss")
    plt.legend(loc="upper right")
    plt.xlim([0, 30])
    plt.ylim([0, 1])
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.savefig('graphen/Epoch and Loss.png', dpi=100)
    plt.close()


# Function to print accuracy 
def accuracy(p_valaccuracy, p_trainaccuracy, p_epoch):
    plt.title('Accuracy and Epoch')
    plt.plot(p_epoch,p_valaccuracy, color='orange', label="val accuracy")
    plt.plot(p_epoch,p_trainaccuracy, "-b", label="train accuracy")
    plt.legend(loc="upper right")
    plt.xlim([0, 30])
    plt.ylim([0, 1.1])
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.savefig('graphen/Epoch and Accuracy.png', dpi=300)
    plt.close()


# Function to print lengths
def lengths(lengths):

    fig, ax = plt.subplots()


    num_bins = 60
    # the histogram of the data
    n, bins, patches = ax.hist(lengths, num_bins)
    
    ax.set_xlabel('Length')
    ax.set_ylabel('Amount')
    ax.set_title(r'Histogram of lengths')

    # Tweak spacing to prevent clipping of ylabel
    fig.tight_layout()
    plt.savefig('graphen/lengths.png', dpi=300)
    plt.close()



# Function to print ROC and AUC 
from sklearn.metrics import accuracy_score, roc_curve, auc

def evaluate_roc(probs, y_true):
    """
    - Print AUC and accuracy on the test set
    - Plot ROC
    @params    probs (np.array): an array of predicted probabilities with shape (len(y_true), 2)
    @params    y_true (np.array): an array of the true values with shape (len(y_true),)
    """
    preds = probs[:, 1]

    fpr, tpr, threshold = roc_curve(y_true, preds)
    roc_auc = auc(fpr, tpr)
    print(f'AUC: {roc_auc:.4f}')
       
    # Get accuracy over the test set
    y_pred = np.where(preds >= 0.5, 1, 0)
    accuracy = accuracy_score(y_true, y_pred)
    print(f'Accuracy: {accuracy*100:.2f}%')
    
    # Plot ROC AUC
    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.savefig('graphen/ROC AUC.png', dpi=300)
    plt.close()


# Function to calculate final score of seperated predction tasks 
def final_score(probs_school_achievements, probs_professional_education, probs_skills, scoretype):

    # Unbiased score 
    if scoretype == "unbiased":
        final_score = probs_professional_education + probs_school_achievements + probs_skills

    # Put bias on prediction results of the categories and save the sum 
    if scoretype == "heuristic":
        final_score = (probs_professional_education * 0.6)  + (probs_school_achievements * 0.3) + (probs_skills * 0.1)

    # One hot encoding to choose prediction for label with highest probability 
    final_score = np.argmax(final_score, axis=1)

    for idx, x in enumerate(final_score): 
        if x == 1:
            print("Resume ", idx, " classified as: ")
            print("Predicted label type: 1 -> rejection\n")
            final_score = 1
        else:
            print("Resume ", idx, " classified as: ")
            print("Predicted label type: 0 -> acceptance\n")
            final_score = 0

    return final_score



# Function to one hot encode score of predictions
def single_score(probs):
    print(probs)
    probs_OHE = np.argmax(probs, axis=1)

    i = 1
    for x in probs_OHE: 
        if x == 1:
            print("Resume ", i, " classified as: ")
            print("Predicted label type: 1 -> rejection\n")
            probs_OHE = 1
            
        else:
            print("Resume ", i, " classified as: ")
            print("Predicted label type: 0 -> acceptance\n")
            probs_OHE= 0

        i=i+1

    return probs_OHE
