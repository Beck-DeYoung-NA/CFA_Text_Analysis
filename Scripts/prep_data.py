
import pandas as pd
from bs4 import BeautifulSoup
import re
import copy
from typing import List

def load_data(question: int, date: str) -> pd.DataFrame:
    # Read in the data for the given question.
    q_data_path = f'../Data/NAXION_CFA_OrgDrag_codingexamples_{date}_Q{question}.csv'
    q_data = pd.read_csv(q_data_path, encoding='latin1')
    q_data = q_data.rename(columns={q_data.columns[1]: "Body"}) # Rename second column to Body
    
    # Codes are loaded in as floats, so this will convert them all to integers
    float_cols = q_data.filter(regex='^acode').columns
    q_data[float_cols] = q_data[float_cols].fillna(0).astype(int)
    # q_data.index.names = ['Id'] # Create a unqiue index based on the row number
    q_data = q_data[q_data['acode1'] < 98].reset_index()
    return q_data
    
def get_top_tags(df: pd.DataFrame, num_codes: int) -> set:
    # Get the unique codes from each respondent and combine them into a single list, ignoring missing values
    all_acodes = df.filter(regex='acode.*').values.flatten()
    all_acodes = all_acodes[all_acodes != 0].tolist() # ~np.isnan(all_acodes)
    top_codes = (pd.DataFrame({'acode': all_acodes})
                    .dropna()
                    .groupby('acode').size().reset_index(name='n')
                    .query(f"acode not in [98, 99]")
                    .sort_values(by='n', ascending=False).head(num_codes))
    return set(top_codes['acode'])

def create_top_codes_col(row: pd.Series, top_codes_set: set) -> List[int]:
    # Use set intersection to find the top codes that are present in the row
    top_codes_list = list(top_codes_set.intersection(row.filter(regex='acode.*').values))
    # Return the top_codes_list
    return top_codes_list

def clean_body(text: str) -> str:
  text = BeautifulSoup(text, 'html.parser').get_text()
  tokens = re.sub("[^a-zA-Z]", " ", text).lower().split()
  return " ".join(tokens)
    
def prep_question(df: pd.DataFrame, num_tags: int) -> pd.DataFrame:
    df = copy.deepcopy(df)
    top_codes_set = get_top_tags(df, num_tags)
    # Create the tags column and filter for only bodies with top tags
    df['Tags'] = df.apply(create_top_codes_col, top_codes_set=top_codes_set, axis=1)
    df = df[df['Tags'].apply(len) > 0]
    # Clean up the body column
    df["Body"] = df['Body'].apply(clean_body)
    
    return df[["Body", "Tags"]]