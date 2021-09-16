import data_preparation
import pdf_to_txt
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import decimal
import glob
import yaml
import os

# loading config params
project_root =  "/mnt/md0/user/swidnickira68812/bertv1"
with open(project_root+"/config.yml") as f:
    params = yaml.load(f, Loader=yaml.FullLoader)

path_files = params["textsimilarity"]["path"]

#Save each resume from .txt file as a sentence to one list 
def get_sentences(path):

    myFilesPaths = sorted(glob.glob(path+'*.txt'))
    print("\n Found .txt data for similarity check: \n")
    print(myFilesPaths)

    sentences = []

    for x in myFilesPaths:        
        with open(x, 'r', encoding="utf8", errors='ignore') as file:
            data = file.read().replace('\n', ' ')
        sentences.append(data)

    return sentences

#Check similarity for the list of sentences (resumes)
def find_similarity(sentences):
    
    sentences_short = []

    for x in sentences:
        x = data_preparation.text_preprocessing(x)
        sentences_short.append(x)

    # Find the maximum length
    max_len = max([len(sent) for sent in sentences_short])
    print('\n Max length: ', max_len)

    tokenizer = AutoTokenizer.from_pretrained(params["textsimilarity"]["model"])
    model = AutoModel.from_pretrained(params["textsimilarity"]["model"])

    # initialize dictionary to store tokenized sentences
    tokens = {'input_ids': [], 'attention_mask': []}

    for sentence in sentences_short:
        # encode each sentence and append to dictionary
        new_tokens = tokenizer.encode_plus(sentence, max_length=512,
                                        truncation=True, padding='max_length',
                                        return_tensors='pt')
        tokens['input_ids'].append(new_tokens['input_ids'][0])
        tokens['attention_mask'].append(new_tokens['attention_mask'][0])

    # reformat list of tensors into single tensor
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    outputs = model(**tokens)
    embeddings = outputs.last_hidden_state
    attention_mask = tokens['attention_mask']
    mask = attention_mask.unsqueeze(-1).expand(embeddings.size()).float()
    masked_embeddings = embeddings * mask
    summed = torch.sum(masked_embeddings, 1)
    summed_mask = torch.clamp(mask.sum(1), min=1e-9)
    mean_pooled = summed / summed_mask

    # convert from PyTorch tensor to numpy array
    mean_pooled = mean_pooled.detach().numpy()

    # calculate
    results = cosine_similarity([mean_pooled[0]], mean_pooled[1:])

    return results
    

#Sort results by decreasing suitability of resumes and displays sorted order
def show_results(results):

    # Order results by decreasing percentage 
    idx_results_sorted = np.argsort(-results)
    probs_results_sorted = -np.sort(-results)

    #### Section to convert idx_results_sorted to iterable list  ####

    # Convert results to list
    a = np.array(idx_results_sorted).tolist()
    #Create string from list
    results_list = ''.join(str(e) for e in a)

    list_idx_sorted = []
    x = " "

    if len(idx_results_sorted) > 0:
    # Create list of results through splitting string 
  
        for index in results_list:
            if index == "," or index == "]":
                x.strip()
                list_idx_sorted.append(x)
                x = ""
            
            if index != "," and index != "[":
                x = x+str(index)
    # Incase only one resume is part of the list
    else:
        results_list = results_list.replace('[', '')
        list_idx_sorted.append(results_list.replace(']', ''))

    #### Section to convert probs_results_sorted to iterable list #### 

    # Convert results to list
    a = np.array(probs_results_sorted).tolist()
    #Create string from list
    results_list = ''.join(str(e) for e in a)
    
    list_probs_sorted = []
    x = " "

    if len(idx_results_sorted) > 0:
        # Create list of results through splitting string 
        for index in results_list:
            if index == "," or index == "]":
                x.strip()
                list_probs_sorted.append(x)
                x = ""
            
            if index != "," and index != "[":
                x = x+str(index)               
    # Incase only one resume is part of the list
    else:
        results_list = results_list.replace('[', '')
        list_probs_sorted.append(results_list.replace(']', ''))

    #### Section to print results #### 
    print("\n Die folgende Auflistung ist absteigend: \n")

    for x, y in zip(list_idx_sorted, list_probs_sorted):
        print("Lebenslauf", str(int((x))+1), " hat eine Ähnlichkeit zur Stellenausschreibung von: ", str(round(decimal.Decimal((y))*100, 2)), " Prozent")

    return list_idx_sorted


def sort_pdf(idx_order):

    filepaths = []
    sorted_list = []

    #Create list with all resume files 
    for x in glob.glob(path_files+'*.pdf'):
        filepaths.append(x)

    print("\n Found pdf data for similarity check: \n")
    print(filepaths)

    #Create list with sorted resume pdf filenames
    for resume_number in idx_order:
        sorted_list.append(filepaths[int(resume_number)])
        print(filepaths[int(resume_number)])

    #Rename the pdf files to sort them inside the directory
    for idx, resume in enumerate(sorted_list):
        changed_name = path_files +str(idx+1) + "_" + resume.rsplit('/', 1)[-1]
        os.rename(filepaths[idx], changed_name)

    print("\n - - - Resumes have been sorted by decreasing similarity to job description")

#pdf_to_txt.convert_pdf(path_files)
show_results(find_similarity(get_sentences(path_files)))

