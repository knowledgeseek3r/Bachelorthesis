import torch
import torch.nn as nn
from transformers import BertModel
import yaml
from torchsummary import summary

# loading config params
project_root =  "/mnt/md0/user/swidnickira68812/bertv1"
with open(project_root+"/config.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

#-------------------------------       --------------------------------
#------------------------------- MODEL --------------------------------
#-------------------------------       --------------------------------

#Bert V1
path_fullmodel = params["model"]["modelname"]
#Bert V2 school
path_v2school= params["model"]["path_v2school"]
#Bert V2 profession
path_v2profession = params["model"]["path_v2profession"]
#Bert V2 skills
path_v2skills = params["model"]["path_v2skills"]

# Create the BertClassfier class
class BertClassifier(nn.Module):
 
    """Bert Model for Classification Tasks.
    """
    def __init__(self, freeze_bert, modelname):
        """
        @param    bert: a BertModel object
        @param    classifier: a torch.nn.Module classifier
        @param    freeze_bert (bool): Set `False` to fine-tune the BERT model
        """
        super(BertClassifier, self).__init__()
        # Specify hidden size of BERT, hidden size of our classifier, and number of labels
        D_in, H, D_out = 768, 50, 2

        #load the Model
        self.bert = BertModel.from_pretrained(path_fullmodel, local_files_only=True)
        
        #Optional model invoke 
        if modelname == "school":
            self.bert = BertModel.from_pretrained(path_v2school, local_files_only=True)
        if modelname == "profession":
            self.bert = BertModel.from_pretrained(path_v2profession, local_files_only=True)
        if modelname == "skills":
            self.bert = BertModel.from_pretrained(path_v2skills, local_files_only=True)


        # Instantiate an one-layer feed-forward classifier
        self.classifier = nn.Sequential(
            nn.Linear(D_in, H),
            nn.ReLU(),
            nn.Dropout(params["model"]["dropout"]),
            nn.Linear(H, D_out)
        )

        #Freeze the BERT model
        if freeze_bert == True:
            for param in self.bert.parameters():
                param.requires_grad = False
    
    def forward(self, input_ids, attention_mask):
        """
        Feed input to BERT and the classifier to compute logits.
        @param    input_ids (torch.Tensor): an input tensor with shape (batch_size,
                      max_length)
        @param    attention_mask (torch.Tensor): a tensor that hold attention mask
                      information with shape (batch_size, max_length)
        @return   logits (torch.Tensor): an output tensor with shape (batch_size,
                      num_labels)
        """
        # Feed input to BERT
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        
        # Extract the last hidden state of the token `[CLS]` for classification task
        last_hidden_state_cls = outputs[0][:, 0, :]

        # Feed input to classifier to compute logits
        logits = self.classifier(last_hidden_state_cls)

        return logits