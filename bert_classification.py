
### 1. Import required packages

from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, AutoConfig,BertTokenizer,BertModel
from torch.utils.data import DataLoader
from transformers import AdamW
import pandas as pd
import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from tqdm import tqdm
import json
import nltk
from collections import OrderedDict
#import logging
#logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

### 2. Apply tokenizer，wordpiece

# use the bert-base tokenizer while the model is lowercase processed
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

### 3.Read data input

train_data = json.load(open("genre_train.json", "r"))
texts = train_data['X']
labels = train_data['Y'] # id mapping: 0 - Horror, 1 - Science Fiction, 2 - Humor, 3 - Crime Fiction

# load the test data
# the test data does not have labels, our model needs to generate these
test_data = json.load(open("genre_test.json", "r"))
Xt = test_data['X']

# Delete all sub-paragraph identifiers '\n' from the text
def process_tokens(doc):
    doc = doc[1:-1].replace('\n','')
    return doc

texts = [process_tokens(i) for i in texts]

print('length of the data text:',len(texts))

### 4.split train and validation set

# use 10% data to be the validation set and used to do fine-tuning.
  
train_texts, val_texts, train_labels, val_labels = train_test_split(
    texts, labels, test_size=0.1, random_state=42, 
     stratify=labels)   # stratify = lables, allocated according to the proportion in labels


def count_labels(y):
    humors=0
    science=0
    crime =0 
    horror =0
    for i in range(len(y)):
        if labels[i]==0:
            horror+=1
        elif labels[i]==1:
            science+=1
        elif labels[i]==2:
            humors+=1
        else:
            crime+=1
    return horror,science,humors,crime

# equals to the original distribution in labels. We can see distribution is unbalanced.
print('label distribution in the validation set',count_labels(val_labels))

### 5.Find the maximum length for tokenizer

# find the longest sentence in the trian set and validiation set.
max_sentence = max([len(item) for item in train_texts])
print('max length in train_text', max_sentence)

max_sentence = max([len(item) for item in val_texts])
print('max length in val_text',max_sentence)

### 6. maping between labels and id.

label_to_id = OrderedDict(
    {item: idx
     for idx, item in enumerate(set(train_labels + val_labels))})

id_to_label = OrderedDict({v: k for k, v in label_to_id.items()})

### 7. tokenizer for train and validation set.

# padding the text tokenizer list to be the same length = 265, 

train_encodings = tokenizer(train_texts,
                            truncation=True,
                            padding='max_length',
                            max_length=256)
val_encodings = tokenizer(val_texts,
                          truncation=True,
                          padding='max_length',
                          max_length=256)

### 8.create new Dataset

# combine encodings with labels to form new data set
class CuDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        idx = int(idx)
        item = {
            key: torch.tensor(val[idx])
            for key, val in self.encodings.items()
        }
        item['labels'] = torch.tensor(label_to_id[self.labels[idx]])
        return item

    def __len__(self):
        return len(self.labels)
    
# get new train and validation dataset.
train_dataset = CuDataset(train_encodings, train_labels)
val_dataset = CuDataset(val_encodings, val_labels)

### 9. create Dataloader

# train the model using batch_size =32. 
batch_size = 32
# use Dataloader to load the batch of data.
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
eval_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

### 10.Load the model

device =torch.device('cuda:0') if torch.cuda.is_available() else torch.device(
    'cpu')     # use CPU in my own computer, use GPU in the rented server
               # I rented a server to training the model. For faster taining speed.

# apply pre-trained BERT model
model = AutoModelForSequenceClassification.from_pretrained(
    "bert-base-uncased", num_labels=len(label_to_id))

model.to(device)
model.train()

### 11. calculate Accuracy，Precision，Recall，F1 score，confusion_matrix，classification_report

def compute_metrics(labels, preds):
    '''compare the ground truth labels and model predictions,
    use confusion matric to evaluate the model, since the labels are imbalanced.'''
    
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='macro')
    recall = recall_score(labels, preds, average='macro')
    f1 = f1_score(labels, preds, average='macro')
    print(f'accuracy: {accuracy}\n')
    print(f'precision: {precision}\n')
    print(f'recall: {recall}\n')
    print(f'f1: {f1}\n')
    print(confusion_matrix(labels, preds))
    print(classification_report(labels,preds))
    
### 12. Evaluate the model

def eval(model, eval_loader):
    '''evaluate current model,compare the predicted labels with the ground truth labels '''
    
    model.eval()
    labels = []
    preds = []
    for idx, batch in enumerate(eval_loader):
        input_ids = batch['input_ids'].to(device)
        labels.extend(batch['labels'].numpy())
        outputs = model(input_ids)  # outputs are probabilities for 4 labels
        preds.extend(torch.argmax(outputs[0], dim=-1).cpu().numpy())  # use argmax to get the maximum probability label.
    compute_metrics(labels, preds) # Accuracy，Precision，Recall，F1 score，confusion_matrix，classification_report 
    model.train()
    return preds

### 13. Train the model

optim = torch.optim.Adam(model.parameters(), lr=0.0001)  # define optimizer and learning rate 
step = 0  
for epoch in range(10):
    #for idx, batch in tqdm(enumerate(train_loader),  # use tqdm to show the Progress Alert Message
    #                       total=len(train_texts) // batch_size):
    for idx, batch in enumerate(train_loader):  # with out tqdm (if we can not import tqdm)
        optim.zero_grad()
        input_ids = batch['input_ids'].to(device)
        labels = batch['labels'].to(device)
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs[0]  # compute Loss
       # logging.info(
           # f'Epoch-{epoch}, Step-{step}, Loss: {loss.cpu().detach().numpy()}') 
           # use logging to show the training process and loss.  comment it since we can't import logging. 
        step += 1
        loss.backward()
        optim.step()
    print(f'Epoch {epoch}, start evaluating.')
    preds = eval(model, eval_loader)  # evaluate the model
    model.save_pretrained(f'model_{epoch}')  # save the current model.
    tokenizer.save_pretrained(f'model_{epoch}') # save the current tokenizer.
    

def predict(model, tokenizer, text):
    ''' use input model predict the label of the text as output'''
    
    encoding = tokenizer(text,
                         return_tensors="pt",
                         max_length=256,
                         truncation=True,
                         padding=True)["input_ids"]
    outputs = model(encoding)
    pred = id_to_label[torch.argmax(outputs[0], dim=-1).cpu().numpy()[0]]
    return pred

### 14. Find best model

# after tuning hyperparameters, use the selected best model to do final prediction
tokenizer = AutoTokenizer.from_pretrained("model_8") # choose model_8 since epoch 8 has the best performance.
model = AutoModelForSequenceClassification.from_pretrained(
    "model_8", num_labels=len(label_to_id))

### 15. write the output csv file.
fout = open("out.csv", "w")
fout.write("Id,Y\n")
for idx,x in enumerate(Xt):
    pre = predict(model,tokenizer,process_tokens(x))
    fout.write("%d,%d\n" % (idx, pre))
fout.close()