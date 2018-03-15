### Comptez les occurences mots
### les trié puis les afficher
###  

import csv, sys
Text=[]
Poid=[]
TexPoid=[]
Mot=[]
Occurence=[]
taille=int(input("Entrez le nombre de ligne que vous voulez (1 à 10000) : "))
while(taille<1 or taille>10000):
	taille=int(input("Entrez le nombre de ligne que vous voulez (1 à 10000) : "))

with open("dataset.csv") as csvfile:
	#Text=csvfile.read().split(' \n')
	for i in range(taille):
		Text+=[csvfile.readline()]

with open("labels.csv") as csvfile:
	for i in range(taille):
		Poid+=[csvfile.readline()]

TexPoid= zip(Text,Poid)


for i in range(taille):
	print(Text[i],Poid[i])