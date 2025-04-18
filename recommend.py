import numpy as np
import pandas as pd
import ast

movies = pd.read_csv('./datasets/tmdb_5000_movies.csv')
credits = pd.read_csv('./datasets/tmdb_5000_credits.csv')

# print(movies.head())
# print(credits.head())

movies = movies.merge(credits, on='title')
# print(movies.head())

movies = movies[['movie_id', 'title', 'overview', 'genres', 'keywords', 'cast', 'crew']]

# print(movies.duplicated().sum())
movies.dropna(inplace=True) 
# print(movies.isnull().sum())

def convert(obj):
    L=[]
    for i in ast.literal_eval(obj):
        L.append(i['name'])
    return L

def convert3(text):
    L = []
    counter = 0
    for i in ast.literal_eval(text):
        if counter < 3:
            L.append(i['name'])
        counter+=1
    return L 

def fetch_director(text):
    L = []
    for i in ast.literal_eval(text):
        if i['job'] == 'Director':
            L.append(i['name'])
    return L 

def collapse(L):
    L1 = []
    for i in L:
        L1.append(i.replace(" ",""))
    return L1


movies['genres'] = movies['genres'].apply(convert)
movies['keywords'] = movies['keywords'].apply(convert)
movies['cast'] = movies['cast'].apply(convert)
movies['cast'] = movies['cast'].apply(lambda x:x[0:3])
movies['crew'] = movies['crew'].apply(fetch_director)
movies['cast'] = movies['cast'].apply(collapse)
movies['crew'] = movies['crew'].apply(collapse)
movies['genres'] = movies['genres'].apply(collapse)
movies['keywords'] = movies['keywords'].apply(collapse)
movies['overview'] = movies['overview'].apply(lambda x:x.split())
movies['tags'] = movies['overview'] + movies['genres'] + movies['keywords'] + movies['cast'] + movies['crew']
new = movies.drop(columns=['overview','genres','keywords','cast','crew'])
new['tags'] = new['tags'].apply(lambda x: " ".join(x))
print(new.head())
