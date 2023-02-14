# %%
import pandas as pd
import numpy as np
import re
import os

from bert_classes import QTagClassifier, QTagDataset, QTagDataModule
from prep_data import load_data, prep_question, clean_body

QUESTION = 5
NUM_CODES = 13
DATE = "073022"
q_data = load_data(QUESTION, DATE)
all_data, df = prep_question(q_data, NUM_CODES)

df_questions = df[['Id', "Body"]].drop_duplicates()[1:5000]
df_tags = df[["Id", "Tag"]][1:5000]

top_tags = df_tags['Tag'].value_counts().keys()[0:15]
# First group tags Id wise
df_tags = df_tags.groupby('Id').apply(lambda x:x['Tag'].values).reset_index(name='tags')
# %%
# clean the text in Body column
df_questions['Clean_Body'] = df_questions['Body'].apply(clean_body)

# merge tags and questions datasets
df = pd.merge(df_questions,df_tags,how='inner',on='Id')