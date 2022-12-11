#!/usr/bin/env python
# coding: utf-8

# In[66]:


# import packages for nlp
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk import ngrams
from nltk import FreqDist
from collections import Counter


# In[67]:


# open the innovation list txt file (in python directory)
with open('innovation.txt') as f:  
    words500 = f.read()
    print(words500)


# In[68]:


# open a new draft txt file (in python directory)
with open('article.txt') as f:   
    draft = f.read()
    print(draft)


# In[69]:


# define a new stopwords list
stop_words = stopwords.words('english')
# contain certain stopwords in list
# because they are in some phrases of the 500 words list 
non_stop = ['and','of','to','your','about']   
new_stop_words = [word for word in stop_words if word not in non_stop]
print(new_stop_words)


# In[70]:


# tokenize the draft article
new_draft = word_tokenize(draft)   
# make all lower cases
new_draft = [t.lower() for t in new_draft]  
# remove updated stopwords
new_draft = [t for t in new_draft if t not in new_stop_words]
# only contain alphabet
new_draft = [t for t in new_draft if t.isalpha()]
# lemmatizes the words
lemmatizer = WordNetLemmatizer()
new_draft = [lemmatizer.lemmatize(t) for t in new_draft]

print(new_draft)


# In[71]:


# tokenize 500 words list
new_words500 = word_tokenize(words500)
print(new_words500)


# In[72]:


# define a function that returns ngrams
def get_ngrams(text, n ):
    n_grams = ngrams(text, n)
    return [ ' '.join(grams) for grams in n_grams]


# In[73]:


# store 4-word grams in 500 list
n4_words500 = get_ngrams(new_words500, 4)


# In[74]:


# store 4-word grams in article draft
n4_new_draft = get_ngrams(new_draft, 4)


# In[77]:


# import and load the spaCy package
import spacy        
nlp = spacy.load('en')

#import phrased-based matching package
from spacy.matcher import PhraseMatcher
matcher = PhraseMatcher(nlp.vocab)


# In[78]:


# make article draft a string again
# to be able to perform spaCy package
str_filtered_draft = ' '.join([str(ele) for ele in n4_new_draft]) 
print(str_filtered_draft) 


# In[79]:


## define patterns and find matches
# make processed Doc for 500 words list
patterns = [nlp(text) for text in n4_words500]   
# add matcher word
matcher.add('TerminologyList', patterns)  
# make processed Doc for article draft
doc = nlp(str_filtered_draft)  
# make matches
matches = matcher(doc)   
# create a new list to print the text string of all matched words
matched_list = [doc[start:end] for match_id, start, end in matches]   
# print the matched list    
print(matched_list)


# In[80]:


# count word frequency in list
frequency_distribution = FreqDist(matched_list)
print(frequency_distribution) 


# In[81]:


# count instances of all matches
frequency_distribution.most_common(1) 


# In[82]:


# In this case, only 1 match of 1 instance for the 4-word grams,
# between the two lists,
# which is "massachusetts institute of technology"


# In[ ]:




