#!/usr/bin/env python
# coding: utf-8

# In[1]:


import nltk
nltk.download()


# In[2]:


from nltk.corpus import names


# In[5]:


def gender_features(word):
    return{'last_letter': word[-1]}


# In[7]:


gender_features('sherlock')


# In[11]:


labeled_names = ([(name, 'male') for name in names.words('male.txt')]+[(name, 'female') for name in names.words('female.txt')])


# In[12]:


labeled_names


# In[13]:


import random


# In[14]:


random.shuffle(labeled_names)


# In[15]:


labeled_names


# In[16]:


featuresets = [(gender_features(n),gender) for (n,gender) in labeled_names]


# In[17]:


featuresets


# In[21]:


train_set, test_set = featuresets[500:], featuresets[:500]


# In[22]:


classifier = nltk.NaiveBayesClassifier.train(train_set)


# In[23]:


classifier.classify(gender_features('David'))


# In[24]:


print(nltk.classify.accuracy(classifier, test_set))


# In[ ]:




