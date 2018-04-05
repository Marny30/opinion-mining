#!/usr/bin/python3

from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score

#ref : http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html

DATASET_PATH = "./Données/dataset.csv"
LABELS_PATH = "./Données/labels.csv"
SHUFFLED_DATASET_PATH = "./Données/dataset_shuffled.csv"
SHUFFLED_LABELS_PATH = "./Données/labels_shuffled.csv"

SHUFFLED_LEMMA_DATASET_PATH = "./Données/lemmatized_dataset_shuffled.csv"

def getLabels(path):
    return [int(line.strip('\n')) for line in open(path)]

def getDataset(path):
    # path = "./datasetlematise.csv"
    return [line.strip('\n') for line in open(path)]

class ClassifierWrapper:
    """ Classifieur paramétrable selon le type de classifieur souhaité
    """
    def __init__(self, data, labels, clf):
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

    def visualization(): # TODO
        pass

# note pour plus tard : pour n=3, dataset non lemmatisé plus précis, mais si n augmente cet écart s'amoindri
def main():
    data = getDataset(SHUFFLED_LEMMA_DATASET_PATH)
    labels = getLabels(SHUFFLED_LABELS_PATH)

    n = 5000
    data_train = data[:n]       # les n premiers
    labels_train = labels[:n]
    data_test = data[n:]        # les n derniers
    labels_test = labels[n:]    
    
    classifier = KNeighborsClassifier(n_neighbors=4)
    k_neighbors_classifier = ClassifierWrapper(data_train, labels_train, classifier)
    prediction = k_neighbors_classifier.predict(data_test)
    
    print("Précision pour kneighbors (n=3) : ", accuracy_score(labels_test, prediction))

if __name__ == '__main__':
    main()
