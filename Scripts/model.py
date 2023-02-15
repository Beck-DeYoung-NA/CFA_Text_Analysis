# %%
# Load the TensorBoard notebook extension
# %load_ext tensorboard
import pandas as pd
import numpy as np
import torch
import matplotlib.pyplot as plt
import re
import os

from transformers import BertTokenizer
import transformers
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader,Dataset,RandomSampler, SequentialSampler

from bert_classes import QTagClassifier, QTagDataset, QTagDataModule
from prep_data import load_data, prep_question
from analysis import plot_word_freq

from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split

# %%
RANDOM_SEED = 42
pl.seed_everything(RANDOM_SEED, workers=True)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

QUESTION = 5
NUM_CODES = 13
DATE = "073022"
N_EPOCHS = 2
BATCH_SIZE = 300
MAX_LEN = 300
LR = 2e-05
BERT_MODEL_NAME = "bert-base-cased"
MAX_WORD_COUNT = 300
# %%
q_data = load_data(QUESTION, DATE)
df = prep_question(q_data, NUM_CODES)
x = df['Body'].to_list()
y = df['Tags'].to_list()

print(f'Num samples for training and testing: {len(x)}')
# %%
mlb = MultiLabelBinarizer()
yt = mlb.fit_transform(y)
# %%
if False: plot_word_freq(x)
# %%
x_train, x_test = x, x
y_train, y_test = yt, yt
x_tr,x_val,y_tr,y_val = train_test_split(x_train, y_train, test_size=0.3, random_state=RANDOM_SEED,shuffle=True)
# %%
# Initialize the Bert tokenizer
Bert_tokenizer: BertTokenizer = BertTokenizer.from_pretrained(BERT_MODEL_NAME)
def tokenize_encode(sentence):
    return Bert_tokenizer.encode(sentence, add_special_tokens=True, max_length=512, truncation=True)

qs_above_max = len([q for q in x if len(tokenize_encode(q)) > MAX_WORD_COUNT])
print(f'# Question having word count > {MAX_WORD_COUNT}: is  {qs_above_max}')

# %%

# Instantiate and set up the data_module
QTdata_module = QTagDataModule(x_tr,y_tr,x_val,y_val,x_test,y_test,
                               Bert_tokenizer,BATCH_SIZE,MAX_LEN)
QTdata_module.setup()

# %%
steps_per_epoch = len(x_tr)//BATCH_SIZE
model = QTagClassifier(n_classes=13, steps_per_epoch=steps_per_epoch,n_epochs=N_EPOCHS,lr=LR)
#Initialize Pytorch Lightning callback for Model checkpointing

checkpoint_callback = ModelCheckpoint(
    monitor='val_loss',# monitored quantity
    filename='QTag-{epoch:02d}-{val_loss:.2f}',
    save_top_k=3, #  save the top 3 models
    mode='min', # mode of the monitored quantity  for optimization
)

# %%
trainer = pl.Trainer(max_epochs = N_EPOCHS ,accelerator = 'cpu', callbacks=[checkpoint_callback], log_every_n_steps=10)
# Train the Classifier Model
# %%
print('training')
trainer.fit(model, QTdata_module)
# Evaluate the model performance on the test dataset
# %%
trainer.test(model,datamodule=QTdata_module)
# Visualize the logs using tensorboard.
# %load_ext tensorboard
# %tensorboard --logdir lightning_logs/
#Evaluate Model Performance on Test Set
# Retreive the checkpoint path for best model
model_path = checkpoint_callback.best_model_path
print(model_path)
len(y_test), len(x_test)
#setup test dataset for BERT