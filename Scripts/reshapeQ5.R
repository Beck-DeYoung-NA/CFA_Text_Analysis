library(tidyverse)
library(glue) 


args = commandArgs(trailingOnly=TRUE)

if (length(args) < 3) stop("Too few arguements passed. Should pass NUM_CODES, Q_NUM, and DATE")
if (length(args) > 3) stop("Too many arguements passed. Should pass NUM_CODES, Q_NUM, and DATE")

NUM_CODES <- 13
Q_NUM <- 5
DATE <- "073022" # MMDDYY

# NUM_CODES <- args[1]
# Q_NUM <- args[2]
# DATE <- args[3] # MMDDYY

# This csv file has stacked all of the 3 Q5 questions
mydata <- "./Data/NAXION_CFA_OrgDrag_codingexamples_{DATE}_Q{Q_NUM}.csv" %>% glue() %>% 
  read.csv(fileEncoding="latin1") %>% 
  rename("Body" = 2) # Rename second column to Body

# actual codes in single list
all_acodes <- mydata %>% 
  select(starts_with('acode')) %>% as.matrix() %>% as.vector()

# Get top codes by number of responses
top_acodes <- data.frame(acode = all_acodes) %>% 
  na.omit() %>% # Drop missing values
  group_by(acode) %>% summarise(n = n()) %>% arrange(desc(n)) %>% # Get counts and order by most instances
  filter(!acode %in% c(98, 99)) %>% # Remove codes 98 and 99
  filter(row_number() <= NUM_CODES) # Only keep the top 13


#Note one respondent can give more than one open-end in Q5 so NA ID's are duplicated
#Let's create unique IDs

mydata <- mydata %>% 
  mutate(Id = row_number()) %>% 
  filter(acode1<98)

mydata %>% write_csv("./Data/NAXION_Q{Q_NUM}_True_Scores_all_data_for_scoring.csv" %>% glue())

for (code in top_acodes$acode){
  mydata[, glue("bcode{code}")] <- rowSums(select(mydata, starts_with("acode")) == code, na.rm = T)
  }

mydata %>% write_csv("./Data/NAXION_Q{Q_NUM}_True_Scores_all_top{NUM_CODES}codes.csv" %>% glue())

# Make longer data set
longdata <- mydata %>% pivot_longer(cols = starts_with("bcode"), 
                                    names_to = "bcode", 
                                    values_to = "code_present") %>% 
  filter(code_present == 1) %>% # code_present is 1 if the code is detected and 0 if not
  mutate(Tag = str_extract(bcode, "[0-9]{1,}") %>% as.numeric()) %>% 
  select(Id, Body, Tag)

longdata %>% write_csv(file="./Data/NAXION_Q{Q_NUM}_long_top{NUM_CODES}codes.csv" %>% glue())


justtags <-longdata %>% select(Id, Tag)
justtags %>% write_csv("./Data/NAXION_Q{Q_NUM}_top{NUM_CODES}tags_for_BERT.csv" %>% glue())


justtext <- mydata %>% select(Id, Body) %>% 
  filter(Id %in% longdata$Id)

justtext %>% write_csv("./Data/NAXION_Q{Q_NUM}_top{NUM_CODES}text_for_BERT.csv" %>% glue())
