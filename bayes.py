#!/usr/bin/python3


from sklearn import datasets
from sklearn.naive_bayes import GaussianNB


gnb = GaussianNB()
x_train = []
x_test = []
y_train = []

with open("datasetSansPonctuation.csv") as csvfile:
	for ligne in range(1, 9) :
		if(ligne < 6):
			x_train.append(csvfile.readline())
		
		else:
			x_test.append(csvfile.readline())
	#print(corpus[i])
        

with open("labels.csv") as csvfile:
	for ligne in range(1, 6) :
		y_train.append(csvfile.readline())
        



print(x_train)
print(y_train)	
gnb.fit(x_train, y_train)
y_pred= gnb.predict(x_test)

#.predict(corpus)
print(y_pred)
