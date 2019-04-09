#!/usr/bin/env python
# coding: utf-8

# In[6]:


example1="""If there is a phrase I would prefer to retire from online bios, personal and professional, it is, “I love travel.” Or some proximation of that sentiment. To clarify, I’m not against travelers and those who proudly flaunt their passion for travel. On the contrary, editing a travel magazine has now made me oddly protective of travelers and their ilk. My submission is that “love to travel,” suggesting so casually, just doesn’t feel adequate to the depth of emotion it sparks in true devotees"""
example2="""Summer is a charming flirt. Easy-going and casual. Summer doesn.t huff and puff to win our affections. It has us at "Hello". Winter broods like the tortured protagonist of big fat Russian novel. It is dauting and dramatic, burning with a slow intensity.The season's reputation precedes itself, and often, not in a good way. It has a way of whittling down everything to its bare bones. Even relationship not attuned to its ebbs and flows can fray. At a dinner conversation I once attended. I listened in bemusement as a recent divorcee made the case that it was the Scandinavian frost that had cooled his ex-wife's ardor. How original."""
example3="""One of the finer books I read this year was John Kaag’s Hiking With Nietzsche, in which Kaag, a professor of philosophy, rekindles his passion for the German thinker while tracing picturesque hiking trails in the mountains of Switzerland. It’s a near-precise rendering of the travelogue as a self-help book. A young Kaag was an avowed Nietzsche acolyte but given the ravages of responsibilities and adulthood, the writer put his affinity to test by undertaking physically enduring hikes through the Alps, revisiting haunts that the philosopher escaped to, in search of solitude and salve. The journey’s demands, coupled with his own inner turmoil, are catnip for anybody feeling at cross purposes with their own life."""


# In[13]:


from sklearn.feature_extraction.text import TfidfVectorizer
vect = TfidfVectorizer(binary = True)
corpus = [example1,example3]
vect.fit(corpus)


# In[14]:


from sklearn.metrics.pairwise import cosine_similarity
similarity = cosine_similarity(vect.transform([example1]).toarray(), vect.transform([example2]).toarray())
print(similarity)


# In[ ]:




