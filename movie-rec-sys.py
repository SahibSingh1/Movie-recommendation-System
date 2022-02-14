#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd
movies=pd.read_csv('tmdb_5000_movies.csv')
credits=pd.read_csv('tmdb_5000_credits.csv')
movies=movies.merge(credits,on="title")
movies= movies[['title','genres','id','keywords','overview','cast','crew']]


# movies.isnull().sum()
# checked if any movie had any missing data

movies.dropna(inplace=True)
# then deleted those elements(movies) with null values

# movies.iloc[0].genres
#but now the keywords aren't in the way we want them to be

# so now we'll select these keywords
# but the problem will be that the list is actually a string so for that we'll use literal eval of the ast module
import ast
def convert(ar):
    l=[]
    for i in ast.literal_eval(ar):
        l.append(i['name'])
    return l
    
movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)

def converts(ar):
    l=[]
    c=0
    for i in ast.literal_eval(ar):
        l.append(i['name'])
        c+=1
        if c==5:
            break
    return l
#now we chose top 5 cast of the movies

movies['cast']=movies['cast'].apply(converts)

def finddirector(ar):
    l=[]
    for i in ast.literal_eval(ar):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l

movies['crew']=movies['crew'].apply(finddirector)

#the overview is a string but to use is as we like we'll have to make it as a list
movies['overview']=movies['overview'].apply(lambda x:x.split())

movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])

movies['tags']=movies['overview']+movies['genres']+movies['keywords']+movies['cast']+movies['crew']

movies=movies[['id','title','tags']]
movies['tags']=movies['tags'].apply(lambda x:" ".join(x))


# now there are some similar words like dance , danced , dancing which should be the same 
# so for that we'll use stemming
import nltk
from nltk.stem import PorterStemmer
ps=PorterStemmer()

# now we'll stem each word 
def stemming(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)

movies['tags']=movies['tags'].apply(stemming)

from sklearn.feature_extraction.text import CountVectorizer 
cv=CountVectorizer(max_features=5000,stop_words='english')

vector=cv.fit_transform(movies['tags']).toarray()


# now as for measuring the similarity we'll have to calculate the sistance between these vectors in a 5000 dimension
# for which we'll use cosine distance which is and angle
#the lesser the angle the more simlarity


from sklearn.metrics.pairwise import cosine_similarity
similarity=cosine_similarity(vector)

# in this similarity matrix , the calculated similarity of every movies to each other is shown
#thus the shape is squarical
# and the diagonal will always be one as its just its similarity to itself

def rec(movie):
    index=movies[movies['title']==movie].index[0]
    dist=similarity[index]
    movielist=sorted(list(enumerate(dist)),reverse=True,key=lambda x:x[1])[1:11]
    for i in movielist:
        print(movies.iloc[i[0]].title)


# In[ ]:


rec(str(input()))


# In[ ]:




