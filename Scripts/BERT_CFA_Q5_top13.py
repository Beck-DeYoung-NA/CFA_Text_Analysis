# %%
# https://github.com/pnageshkar/NLP/blob/master/Medium/Multi_label_Classification_BERT_Lightning.ipynb
# Import all libraries
import pandas as pd
import numpy as np
import re

# Huggingface transformers
import transformers
from transformers import BertModel,BertTokenizer,AdamW, get_linear_schedule_with_warmup

import torch
from torch import nn ,cuda
from torch.utils.data import DataLoader,Dataset,RandomSampler, SequentialSampler

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

#handling html data
from bs4 import BeautifulSoup

import seaborn as sns
from pylab import rcParams
import matplotlib.pyplot as plt
from matplotlib import rc

from bert_classes import QTagClassifier, QTagDataset, QTagDataModule
from prep_data import prep_question

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %%
q_data, df = prep_question(question = 5, num_codes = 13, date = "073022")

df_questionsA = df[['Id', "Body"]].drop_duplicates()
df_tagsA = df[["Id", "Tag"]]

# %%
# load the questions data set
df_questionsA=pd.read_csv("../Data/NAXION_Q5_top13text_for_BERT.csv", encoding='iso-8859-1')
#load the tags dataset
df_tagsA=pd.read_csv("../Data/NAXION_Q5_top13tags_for_BERT.csv", encoding='iso-8859-1')
# %%
#take a subset of the data so the code runs faster
df_questions=df_questionsA[1:5000]
df_tags=df_tagsA[1:5000]

# df_questions=df_questionsA
# df_tags=df_tagsA
# %%
#Pre process the data
def pre_process(text):

  text = BeautifulSoup(text).get_text()
  
  # fetch alphabetic characters
  text = re.sub("[^a-zA-Z]", " ", text)

  # convert text to lower case
  text = text.lower()

  # split text into tokens to remove whitespaces
  tokens = text.split()

  return " ".join(tokens)

# %%
# clean the text in Body column
df_questions['Clean_Body'] = df_questions['Body'].apply(pre_process)

# %%
top_tags = df_tags['Tag'].value_counts().keys()[0:15]
# First group tags Id wise
df_tags = df_tags.groupby('Id').apply(lambda x:x['Tag'].values).reset_index(name='tags')
# merge tags and questions datasets
df = pd.merge(df_questions,df_tags,how='inner',on='Id')
df.to_csv("../Data/outQ5top13_cleanbody.csv")
# Retain only the columns we will use for training the model - Tags will be the label
df = df[['Clean_Body','tags']]

#df.to_csv("clean_question_tag.csv")
# Filter out records ( values in clean_body and tags) that have atleast one of the top tags
# %%
x, y = [], [] # Filtered body and corresponding tags
lst_top_tags = list(top_tags)

for i, tags in enumerate(df['tags']):
    filtered_tags = [tag for tag in tags if tag in lst_top_tags]
    if filtered_tags:
        x.append(df.at[i, 'Clean_Body'])
        y.append(filtered_tags)
        
# number of records that will be used for training and testing
print(f'Num samples for training and testing: {len(x)}')

# %%

# Encode the tags(labels) in a binary format in order to be used for training

from sklearn.preprocessing import MultiLabelBinarizer
mlb = MultiLabelBinarizer()
 
yt = mlb.fit_transform(y)
yt.shape
# Getting a sense of how the tags data looks like
print(yt[0])
print(mlb.inverse_transform(yt[0].reshape(1,-1)))
print(mlb.classes_)

# %%
# compute no. of words in each question
questions = x
word_cnt = [len(quest.split()) for quest in questions]
# Plot the distribution
plt.figure(figsize=[8,5])
plt.hist(word_cnt, bins = 40)
plt.xlabel('Word Count/Question')
plt.ylabel('# of Occurrences')
plt.title("Frequency of Word Counts/sentence")
plt.show()
# %%
#Split into train, validate and test
#from sklearn.model_selection import train_test_split
# First Split for Train and Test
#x_train,x_test,y_train,y_test = train_test_split(x, yt, test_size=0.2, random_state=RANDOM_SEED,shuffle=True)
# Next split Train in to training and validation
#x_tr,x_val,y_tr,y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=RANDOM_SEED,shuffle=True)
#Use all data for training
#Use all data for scoring
x_train=x
x_test=x
y_train=yt
y_test=yt
x_tr,x_val,y_tr,y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=RANDOM_SEED,shuffle=True)
len(x_tr) ,len(x_val), len(x_test)

# %%

# Initialize the Bert tokenizer
BERT_MODEL_NAME = "bert-base-cased" # we will use the BERT base model(the smaller one)
Bert_tokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
MAX_WORD_COUNT = 300

def tokenize_encode(sentence):
    return Bert_tokenizer.encode(sentence, add_special_tokens=True)

qs_above_max = len([q for q in questions if len(tokenize_encode(q)) > MAX_WORD_COUNT])

print(f'# Question having word count > {MAX_WORD_COUNT}: is  {qs_above_max}')
# %%
# Initialize the parameters that will be use for training
N_EPOCHS = 12
BATCH_SIZE = 32
MAX_LEN = 300
LR = 2e-05
# Instantiate and set up the data_module
QTdata_module = QTagDataModule(x_tr,y_tr,x_val,y_val,x_test,y_test,Bert_tokenizer,BATCH_SIZE,MAX_LEN)
QTdata_module.setup()
#Train the Model
    
# Instantiate the classifier model
steps_per_epoch = len(x_tr)//BATCH_SIZE
model = QTagClassifier(n_classes=13, steps_per_epoch=steps_per_epoch,n_epochs=N_EPOCHS,lr=LR)
#Initialize Pytorch Lightning callback for Model checkpointing

# saves a file like: input/QTag-epoch=02-val_loss=0.32.ckpt
checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',# monitored quantity
    filename='QTag-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3, #  save the top 3 models
    mode='min', # mode of the monitored quantity  for optimization
)
# Instantiate the Model Trainer
trainer = pl.Trainer(max_epochs = N_EPOCHS , callbacks=[checkpoint_callback])
#torch.cuda.memory_summary(device=None, abbreviated=False)
# Train the Classifier Model
trainer.fit(model, QTdata_module)
# Evaluate the model performance on the test dataset
trainer.test(model,datamodule=QTdata_module)
# Visualize the logs using tensorboard.
#%load_ext tensorboard
#%tensorboard --logdir lightning_logs/
#Evaluate Model Performance on Test Set
# Retreive the checkpoint path for best model
model_path = checkpoint_callback.best_model_path
model_path
len(y_test), len(x_test)
#setup test dataset for BERT
from torch.utils.data import TensorDataset

# Tokenize all questions in x_test
input_ids = []
attention_masks = []


for quest in x_test:
    encoded_quest =  Bert_tokenizer.encode_plus(
                    quest,
                    None,
                    add_special_tokens=True,
                    max_length= MAX_LEN,
                    padding = 'max_length',
                    return_token_type_ids= False,
                    return_attention_mask= True,
                    truncation=True,
                    return_tensors = 'pt'      
    )
    
    # Add the input_ids from encoded question to the list.    
    input_ids.append(encoded_quest['input_ids'])
    # Add its attention mask 
    attention_masks.append(encoded_quest['attention_mask'])
    
# Now convert the lists into tensors.
input_ids = torch.cat(input_ids, dim=0)
attention_masks = torch.cat(attention_masks, dim=0)
labels = torch.tensor(y_test)

# Set the batch size.  
TEST_BATCH_SIZE = 64  

# Create the DataLoader.
pred_data = TensorDataset(input_ids, attention_masks, labels)
pred_sampler = SequentialSampler(pred_data)
pred_dataloader = DataLoader(pred_data, sampler=pred_sampler, batch_size=TEST_BATCH_SIZE)
#pred_data[0]
#len(pred_dataloader.dataset)
#Prediction on test set
flat_pred_outs = 0
flat_true_labels = 0
# Put model in evaluation mode
model = model.to(device) # moving model to cuda
model.eval()

# Tracking variables 
pred_outs, true_labels = [], []
#i=0
# Predict 
for batch in pred_dataloader:
    # Add batch to GPU
    batch = tuple(t.to(device) for t in batch)
  
    # Unpack the inputs from our dataloader
    b_input_ids, b_attn_mask, b_labels = batch
 
    with torch.no_grad():
        # Forward pass, calculate logit predictions
        pred_out = model(b_input_ids,b_attn_mask)
        pred_out = torch.sigmoid(pred_out)
        # Move predicted output and labels to CPU
        pred_out = pred_out.detach().cpu().numpy()
        label_ids = b_labels.to('cpu').numpy()
        #i+=1
        # Store predictions and true labels
        #print(i)
        #print(outputs)
        #print(logits)
        #print(label_ids)
    pred_outs.append(pred_out)
    true_labels.append(label_ids)
pred_outs[0][0]
# Combine the results across all batches. 
flat_pred_outs = np.concatenate(pred_outs, axis=0)

# Combine the correct labels for each batch into a single list.
flat_true_labels = np.concatenate(true_labels, axis=0)
flat_pred_outs.shape , flat_true_labels.shape
#Predictions of Tags in Test set
#The predictions are in terms of logits (probabilities for each of the 16 tags). Hence we need to have a threshold value to convert these probabilities to 0 or 1.

#Let's specify a set of candidate threshold values. We will select the threshold value that performs the best for the test set.

#define candidate threshold values
threshold  = np.arange(0.4,0.51,0.01)
threshold
#Let's define a function that takes a threshold value and uses it to convert probabilities into 1 or 0.

# convert probabilities into 0 or 1 based on a threshold value
def classify(pred_prob,thresh):
    y_pred = []

    for tag_label_row in pred_prob:
        temp=[]
        for tag_label in tag_label_row:
            if tag_label >= thresh:
                temp.append(1) # Infer tag value as 1 (present)
            else:
                temp.append(0) # Infer tag value as 0 (absent)
        y_pred.append(temp)

    return y_pred
    
flat_pred_outs[3]
flat_true_labels[3]
flat_pred_outs[20]
flat_true_labels[20]
from sklearn import metrics
scores=[] # Store the list of f1 scores for prediction on each threshold

#convert labels to 1D array
y_true = flat_true_labels.ravel() 

for thresh in threshold:
    
    #classes for each threshold
    pred_bin_label = classify(flat_pred_outs,thresh) 

    #convert to 1D array
    y_pred = np.array(pred_bin_label).ravel()

    scores.append(metrics.f1_score(y_true,y_pred))
# find the optimal threshold
opt_thresh = threshold[scores.index(max(scores))]
print(f'Optimal Threshold Value = {opt_thresh}')
#y_true = flat_true_labels.ravel() 
y_pred
y_true
pd.Series(y_pred).value_counts()
#flat_true_labels
#Performance Score Evaluation

#predictions for optimal threshold
y_pred_labels = classify(flat_pred_outs,opt_thresh)
y_pred = np.array(y_pred_labels).ravel() # Flatten
y_true
y_true.shape
y_pred
y_pred.shape
print(metrics.classification_report(y_true,y_pred))
y_pred = mlb.inverse_transform(np.array(y_pred_labels))
y_act = mlb.inverse_transform(flat_true_labels)

df = pd.DataFrame({'Body':x_test,'Actual Tags':y_act,'Predicted Tags':y_pred})
len(x_test)
len(y_act)
len(y_pred)
x_test
df.shape
df.sample(10)
df.to_csv("U:/computer/Python BERT/CFA data/outQ5top13.csv")
#flat_true_labels.shape
#np.array(y_pred_labels).shape
#y_temp = mlb.inverse_transform(flat_true_labels)
#y_temp
#Inference
# load a model along with its weights, biases and hyperparameters
QTmodel = QTagClassifier.load_from_checkpoint(model_path)
QTmodel.eval()
#Function to Predict Tags from a Question
def predict(question):
    text_enc = Bert_tokenizer.encode_plus(
            question,
            None,
            add_special_tokens=True,
            max_length= MAX_LEN,
            padding = 'max_length',
            return_token_type_ids= False,
            return_attention_mask= True,
            truncation=True,
            return_tensors = 'pt'      
    )
    outputs = QTmodel(text_enc['input_ids'], text_enc['attention_mask'])
    pred_out = outputs[0].detach().numpy()
    #print(f'Outputs = {outputs}')
    #print(f'Type = {type(outputs)}')
    #print(f'Pred Outputs = {pred_out}')
    #print(f'Type = {type(pred_out)}')
    #preds = np.round(pred_out)
    preds = [(pred > opt_thresh) for pred in pred_out ]
    #pred_list = [ round(pred) for pred in pred_logits ]
    preds = np.asarray(preds)
    #print(f'Predictions = {preds}')
    #print(f'Type = {type(preds)}')
    #print(mlb.classes_)
    new_preds = preds.reshape(1,-1).astype(int)
    #print(new_preds)
    pred_tags = mlb.inverse_transform(new_preds)
    #print(mlb.inverse_transform(np.array(new_preds)))
    return pred_tags 
#Try out the Model - Ask a question

# Your question stored in the question variable
question = "based on the following relationship between matthew s correlation coefficient mcc and chi square mcc is the pearson product moment correlation coefficient is it possible to conclude that by having imbalanced binary classification problem n and p df following mcc is significant mcc sqrt which is mcc when comparing two algorithms a b with trials of times if mean mcc a mcc a mean mcc b mcc b then a significantly outperforms b thanks in advance edit roc curves provide an overly optimistic view of the performance for imbalanced binary classification regarding threshold i m not a big fan of not using it as finally one have to decide for a threshold and quite frankly that person has no more information than me to decide upon hence providing pr or roc curves are just for the sake of circumventing the problem for publishing"

# Call the predict function to predict the associated Tags
tags = predict(question)
if not tags[0]:
    print('This Question can not be associated with any known tag - Please review to see if a new tag is required ')
else:
    print(f'Following Tags are associated : \n {tags}')
#save the model
torch.save(QTmodel.state_dict(), 'U:/computer/Python BERT/CFA data/BERT_saved_model/BERT_Q5_top13.pth')
#torch.save(model.state_dict(), 'U:/computer/Python BERT/CFA data/BERT_saved_model/BERT_Q5a.pth')
#saved_model = torch.load('U:/computer/Python BERT/CFA data/BERT_saved_model')

#model=QTagClassifier()
#model.load_state_dict(torch.load('U:/computer/Python BERT/CFA data/BERT_saved_model/BERT_Q5.pth'))
#
#saving model
#https://stackoverflow.com/questions/59340061/saving-a-fine-tuned-bert-model
#https://mccormickml.com/2019/07/22/BERT-fine-tuning/#advantages-of-fine-tuning

# %% 
# %%
import subprocess
res = subprocess.call('Rscript ./reshapeQ5.R 13 5 "073022"', shell=True)
res

# %%
