#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#steps of natural language processing:-


# In[1]:


import nltk
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk import pos_tag, ne_chunk


# In[2]:


nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')


# In[ ]:


text=input("Enter your sentence:")


# In[ ]:


#Step 1: Tokenization
tokens = word_tokenize(text)
print("Step 1: Tokenization")
print(tokens)
print()


# In[ ]:


# Step 2: Stopword Removal
stop_words = set(stopwords.words('english'))
filtered_tokens = [word for word in tokens if word.lower() not in stop_words]
print("Step 2: Stopword Removal")
print(filtered_tokens)
print()


# In[ ]:


# Step 3: Stemming
stemmer = PorterStemmer()
stemmed_tokens = [stemmer.stem(word) for word in filtered_tokens]
print("Step 3: Stemming")
print(stemmed_tokens)
print()


# In[ ]:


# Step 4: Part-of-Speech Tagging
pos_tags = pos_tag(filtered_tokens)
print("Step 4: Part-of-Speech Tagging")
print(pos_tags)
print()


# In[ ]:


# Step 5: Named Entity Recognition (NER)
ner_tags = ne_chunk(pos_tags)
print("Step 5: Named Entity Recognition (NER)")
print(ner_tags)


# In[ ]:




