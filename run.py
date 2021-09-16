import train
import data_preparation
import model
import plot_graph
import matplotlib.pyplot as plt
import yaml
import numpy as np
from sklearn.metrics import confusion_matrix


# loading config params
project_root =  "/mnt/md0/user/swidnickira68812/bertv1"
with open(project_root+"/config.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

########### TRAIND AND VALIDATION DATA SET
#Training Start
train.set_seed(params["training"]["seed"])    # Set seed for reproducibility
model.bert_classifier, train.optimizer, train.scheduler = train.initialize_model(epochs=params["training"]["epochs"])
train.train(model.bert_classifier, data_preparation.train_dataloader, data_preparation.val_dataloader, epochs=params["training"]["epochs"], evaluation=True)


# #Compute predicted probabilities on the test set -> validation
probs = train.bert_predict(model.bert_classifier, data_preparation.val_dataloader)


########### FULL DATA SET TRAINING
# train.set_seed(params["training"]["seed"])    # Set seed for reproducibility
# bert_classifier, train.optimizer, train.scheduler = train.initialize_model(epochs=params["training"]["epochs"])
# train.train(bert_classifier, data_preparation.full_train_dataloader, epochs=params["training"]["epochs"], evaluation=False)

#-------------------------------       --------------------------------
#------------------------------- PLOTTING GRAPHS--------------------------------
#-------------------------------       --------------------------------

#Plotting graphs
plot_graph.evaluate_roc(probs, data_preparation.y_val)
plot_graph.loss(train.p_trainloss, train.p_epoch, train.p_valloss)
plot_graph.accuracy(train.p_valaccuracy, train.p_trainaccuracy, train.p_epoch)

#Compute confusion matrix
probs_fertig = np.around(probs)
probs_fertig = np.argmax(probs_fertig, axis=1)

cnf_matrix = confusion_matrix(data_preparation.y_val, probs_fertig)
np.set_printoptions(precision=2)

# Plot normalized confusion matrix
plt.figure()
plot_graph.confusion_matrix(cnf_matrix, classes=["zusage", "absage"], normalize=True, title='Normalized confusion matrix')
plt.savefig('graphen/Confusion Matrix.png', dpi=300)
plot_graph.bert_measures(cnf_matrix)

#plot_graph.bert_measures(cnf_matrix)
from sklearn.metrics import classification_report
print(classification_report(data_preparation.y_val, probs_fertig))


