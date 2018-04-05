#!/usr/bin/python3

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

#ref : http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

def getLabels():
    path = "./Données/labels.csv"
    return [int(line.strip('\n')) for line in open(path)]

def getDataset():
    # path = "./datasetlematise.csv"
    path = "./Données/dataset.csv"
    return [line.strip('\n') for line in open(path)]

class ClassifierWrapper:
    """ Classifieur paramétrable selon le type de classifieur souhaité
    """
    def __init__(self, clf):
        data = getDataset()
        labels = getLabels()
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
    classifier = KNeighborsClassifier(n_neighbors=3)
    k_neighbors_classifier = ClassifierWrapper(classifier)
    phrasesToTest = ["really cool movie it is awesome.", "the movie sucks"]
    prediction = k_neighbors_classifier.predict(phrasesToTest)
    
    print("Prediction : ")
    for i in range(len(prediction)):
        s = "{:3d}\t{}".format(prediction[i] ,  phrasesToTest[i])
        print(s)

if __name__ == '__main__':
    main()
