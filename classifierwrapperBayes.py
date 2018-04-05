#!/usr/bin/python3

import csv, sys
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score


#ref : http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html


def getDataset(taille):
    Text = []
    with open("lemmatized_dataset_shuffled.csv") as csvfile:
    #Text=csvfile.read().split(' \n')
        for i in range(taille):
            Text+=[csvfile.readline()]
    return Text

def getLabels(taille):
    Poids = []
    with open("labels_shuffled.csv") as csvfile:
        for i in range(taille):
            Poids+=[csvfile.readline()]
    return Poids

def getPredictions(taille):
    Predictions = []
    with open("lemmatized_dataset_shuffled.csv") as csvfile:
        for i in range(10000):
            if i >= taille: 
                Predictions+=[csvfile.readline()]
    return Predictions

def getLabelsVeridique(taille):
	PoidsVrai = []
	with open("labels_shuffled.csv") as csvfile:
		for i in range(10000):
			if i >= taille:
				PoidsVrai+=[csvfile.readline()]
	return PoidsVrai

class ClassifierWrapper:
    """ Classifieur paramétrable selon le type de classifieur souhaité
    """
    def __init__(self, clf, taille):
        data = getDataset(taille)
        labels = getLabels(taille)
        self.count_vect = CountVectorizer()
        data_train_counts = self.count_vect.fit_transform(data)

        # TF-idf : Term Frequency times inverse document frequency
        self.tfidf_transformer = TfidfTransformer()
        data_train_tfidf = self.tfidf_transformer.fit_transform(data_train_counts)
        self.clf = clf
        self.clf.fit(data_train_tfidf, labels)
        
    def predict(self, new_phrases):
        data_new_counts = self.count_vect.transform(new_phrases)
        data_new_tfidf = self.tfidf_transformer.transform(data_new_counts)
        predicted = self.clf.predict(data_new_tfidf)
        return predicted

def main():
    taille=int(input("Entrez le nombre de ligne que vous voulez (1 à 10000) : "))
    mNB = MultinomialNB()
    mb_classifier = ClassifierWrapper(mNB, taille)
    phrasesToTest = getPredictions(taille)
    prediction = mb_classifier.predict(phrasesToTest)
    labelsVrai = getLabelsVeridique(taille)
    print("Prediction : ")
    for i in range(len(prediction)):
    	print(prediction[i] ,  phrasesToTest[i])
    print("Resultats : taux de similarité")
    similarites = accuracy_score(labelsVrai, prediction)
    print(similarites)



if __name__ == '__main__':
    main()


# Resultats : taux de similarité textes bruts
#0.951

#Resultats : taux de similarité textes lemmatisés
#0.926
