#!/usr/bin/env python
# coding: utf-8

# # City Topic Modeling

# ### Loading Dataset

# In[30]:

import gensim
from gensim import corpora
import streamlit as st
import numpy as np
import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer


# #### replace 'path' with the location of the saved dataset after preprocessing, ie. Users/MacbookAir/Desktop/riyadhh.csv

# In[31]:


url="https://raw.githubusercontent.com/Ochie123/Citytopic-modeling/main/Dammam.csv"
my_dataset=pd.read_csv(url)
##s=requests.get(url).content
##my_dataset=pd.read_csv(io.StringIO(s.decode('utf-8')))

# In[32]:


my_dataset.columns


# ## Preprocessing and Cleaning

# ### Lemmatization

# In[33]:


from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize,pos_tag

def lemmatization(text):
    
    result=[]
    wordnet = WordNetLemmatizer()
    for token,tag in pos_tag(text):
        pos=tag[0].lower()
        
        if pos not in ['a', 'r', 'n', 'v']:
            pos='n'
            
        result.append(wordnet.lemmatize(token,pos))
    
    return result


# In[34]:


\


# In[35]:


pd.options.display.max_rows = 9999


# ### Tokenization 

# In[36]:


from nltk import word_tokenize
my_dataset['text']=my_dataset['text'].apply(lambda X: word_tokenize(X))
my_dataset


# In[37]:


from nltk.tokenize import RegexpTokenizer


# In[38]:


def remove_punct(text):
    
    tokenizer = RegexpTokenizer(r"\w+")
    lst=tokenizer.tokenize(' '.join(text))
    return lst


# ### Puntuations Removal

# In[39]:


my_dataset['text'] = my_dataset['text'].apply(remove_punct)
my_dataset


# # Starting our TF-IDF Matrix using gensim library

# In[40]:


import gensim
from gensim import corpora


# In[41]:


dictionary = corpora.Dictionary(my_dataset['text'])


# In[42]:


doc_term_matrix = [dictionary.doc2bow(doc) for doc in my_dataset['text']]
doc_term_matrix


# # Setting up our LDA model

# In[43]:


Lda = gensim.models.ldamodel.LdaModel


# In[44]:


ldamodel = Lda(doc_term_matrix, num_topics=3, id2word = dictionary, passes=50)


# ### Topics Result

# In[45]:


print(ldamodel.print_topics())


# ### From above, the weights associated with the topics are comparable

# ### From above results, we can derive 3 topics from the LDA model


