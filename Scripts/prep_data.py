# %%
import pandas as pd
from bs4 import BeautifulSoup
import re

def load_data(question: int, date: str):
    # Read in the data for the given question.
    q_data_path = f'../Data/NAXION_CFA_OrgDrag_codingexamples_{date}_Q{question}.csv'
    q_data = pd.read_csv(q_data_path, encoding='latin1')
    q_data = q_data.rename(columns={q_data.columns[1]: "Body"}) # Rename second column to Body
    
    # Codes are loaded in as floats, so this will convert them all to integers
    float_cols = q_data.filter(regex='^acode').columns
    q_data[float_cols] = q_data[float_cols].fillna(0).astype(int)
    q_data.index.names = ['Id'] # Create a unqiue index based on the row number
    q_data = q_data[q_data['acode1'] < 98].reset_index(drop=False)
    return q_data
    
def get_top_acodes(df: pd.DataFrame, num_codes: int):
    # Get the unique codes from each respondent and combine them into a single list, ignoring missing values
    all_acodes = df.filter(regex='acode.*').values.flatten()
    all_acodes = all_acodes[all_acodes != 0].tolist() # ~np.isnan(all_acodes)
    return (pd.DataFrame({'acode': all_acodes})
                    .dropna()
                    .groupby('acode').size().reset_index(name='n')
                    .query(f"acode not in [98, 99]")
                    .sort_values(by='n', ascending=False).head(num_codes))
    
def create_dummy_vars(df, top_codes):
    for code in top_codes['acode']:
        df[f'bcode{code}'] = (df.filter(regex='acode.*') == code).sum(axis=1)
    return df

def turn_to_long_format(df):
    longdata = (df.melt(id_vars = ['Id', 'Body'], 
                        value_vars = df.filter(regex="^bcode*").columns,
                        var_name='bcode', value_name='code_present')
                .query(f'code_present == 1')
                .assign(Tag=lambda x: x['bcode'].str.extract(r'(\d+)').astype(int)))
    return longdata[["Id", "Tag", "Body"]]

def prep_question(df: pd.DataFrame, num_codes: int) -> tuple[pd.DataFrame]:
    top_acodes = get_top_acodes(df, num_codes)
    df = create_dummy_vars(df, top_acodes)
    print(df.head(50))
    longdata = turn_to_long_format(df)
    return df, longdata

#Pre process the data
def clean_body(text):

  text = BeautifulSoup(text).get_text()
  
  # fetch alphabetic characters
  text = re.sub("[^a-zA-Z]", " ", text)

  # convert text to lower case
  text = text.lower()

  # split text into tokens to remove whitespaces
  tokens = text.split()

  return " ".join(tokens)

# %%
#mydata.to_csv(f"../Data/NAXION_Q{question}}_True_Scores_all_data_for_scoring.csv", index=False)
#mydata.to_csv(f"../Data/NAXION_Q{question}}_True_Scores_all_top{num_codes}codes.csv", index=False)
#justtags.to_csv(f"../Data/NAXION_Q{question}}_top{num_codes}tags_for_BERT.csv", index=False)
#justtext.to_csv(f"../Data/NAXION_Q{question}}_top{num_codes}text_for_BERT.csv", index=False)]\