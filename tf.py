#!/usr/bin/python3
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()

corpus = []
    
fichier = open("datasetSansPonctuation.csv", "r")
lignes= fichier.readlines()
i=0
for ligne in lignes :
	corpus.append(ligne)
	#print(corpus[i])
	i=i+1
	
fichier.close()

X = vectorizer.fit_transform(corpus)

print(X)


