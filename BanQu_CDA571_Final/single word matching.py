#!/usr/bin/env python
# coding: utf-8

# In[2]:


with open('innovation.txt') as f:   # open the innovation list txt file (in python directory)
    n_words = f.read()
    print(n_words)


# In[3]:


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize

token_n_words = word_tokenize(n_words)     # tokenize innovation list by words 
token_n_words


# In[4]:


with open('draft.txt') as f:   # open the draft txt file (in python directory)
    draft = f.read()
    print(draft)


# In[5]:


token_draft = word_tokenize(draft)    # tokenize draft article by words
token_draft


# In[6]:


# filter out stop words in draft article
from nltk.corpus import stopwords  # import stopwords package from nltk

stop_words = set(stopwords.words("english"))  # focus stopwords in English

filtered_draft = []   # creat an empty list to store the updated content
for word in token_draft:
    if word.casefold() not in stop_words:   # casefold() to ignore if words are uppercase or lowercase
        filtered_draft.append(word)
filtered_draft   # draft list without stop words


# In[7]:


# make filtered draft a string
str_filtered_draft = ' '.join([str(ele) for ele in filtered_draft]) 
str_filtered_draft 


# In[8]:


# import and load the spacy package
import spacy        
nlp = spacy.load('en')

#import phrased-based matching package
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)


# In[9]:


# define patterns and find matches
patterns = [nlp(text) for text in token_n_words]   # make processed Doc for innovation list

matcher.add('TerminologyList', patterns)  # add matcher word:'yes'

doc = nlp(str_filtered_draft)

matches = matcher(doc)   

matched_list = [doc[start:end] for match_id, start, end in matches]   # print the text string of all matched words
    
print(matched_list)


# In[10]:


values = ','.join(str(v) for v in matched_list)   # join all elements in string, seperated by comma
print(values)


# In[11]:


new_list = word_tokenize(values)   # tokenize all words in list
print(new_list)


# In[19]:


# remove punctuations
import string
new_list = [''.join(c for c in s if c not in string.punctuation) for s in new_list]
# remove special characters
special_char = '’'
update_list = [''.join(x for x in string if not x in special_char) for string in new_list]
# remove empty strings
update_list = list(filter(None, update_list))
print(update_list)


# In[20]:


# count word frequency in list
from nltk import FreqDist
frequency_distribution = FreqDist(update_list)
print(frequency_distribution) 


# In[21]:


frequency_distribution.most_common(123)   # count all 123 instances


# In[24]:


frequency_distribution.most_common(123)[:10]  # show top 10 results


# In[ ]:




