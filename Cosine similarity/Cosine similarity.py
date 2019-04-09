#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
from nltk.stem import PorterStemmer
stemmer = PorterStemmer()
example = "The cat was chasing a mouse"
example = [stemmer.stem(token) for token in example.split(" ")]
print(example)


# In[2]:


print(" ".join(example))


# In[4]:


from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
example = "The cat was chasing a mouse"
example = [lemmatizer.lemmatize(token) for token in example.split(" ")]
print(example)


# In[5]:


print(" ".join(example))


# In[6]:


example = (lemmatizer.lemmatize('better', pos = 'a'))


# In[7]:


print(example)


# In[8]:


from sklearn.feature_extraction.text import CountVectorizer


# In[9]:


vect = CountVectorizer(binary = True)
corpus = ["Tesseract is good optical character recognition engine", " optical character recognition is significant"]
vect.fit(corpus)
print(vect.transform(["Today is good optical"]).toarray())


# In[13]:


vect = CountVectorizer(binary = True)
corpus = ["The cat chased a mouse", " mouse is eating chesse"]
vect.fit(corpus)
print(vect.transform(["I am eating chesse"]).toarray())


# In[1]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidVectori(binary = True)
corpus = ["Tesseract is good optical character recognition engine", " optical character recognition is significant"]
vect.fit(corpus)
print(vect.transform(["Today is good optical"]).toarray())


# In[ ]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vect.transform((["Tessaract is an optical character recognition engine"]).toarray, vect.transform(["optical character recognition is significant"]).toarray)
print(similarity)


# In[ ]:





# In[ ]:




