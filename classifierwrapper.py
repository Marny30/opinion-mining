#!/usr/bin/python3

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#ref : http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
class ClassifierWrapper:
    """ Classifieur paramétrable selon le type de classifieur souhaité
    """
    def __init__(self, clf):
        # TF-idf : Term Frequency times inverse document frequency
        self.clf = clf
        # print("entrainement sur " + len(data_train_tfidf) + " données ")
        self.prediction = None
        self.predictionid = None
        
    def train(self, data_train_tfidf, labels):
        self.clf.fit(data_train_tfidf, labels)
        
    def predict(self, data_test_tfdif):
        """ Fonction de prédiction permettant de mémoriser une prédiction déjà
        faite pour le même jeu de donnée
        """
        return self.clf.predict(data_test_tfdif)

def main():
    import utils
    classifier = KNeighborsClassifier(n_neighbors=4)
    # autre exemple
    # classifier = MultinomialNB()
    k_neighbors_classifier = ClassifierWrapper(classifier)
    k_neighbors_classifier.train(utils.FORMATTED_DATA_TRAIN['raw'], utils.LABELS_TRAIN)
    prediction = k_neighbors_classifier.predict(utils.FORMATTED_DATA_TEST['raw'])
    print("Accuracy pour kneighbors (n=3) : ", accuracy_score(utils.LABELS_TEST, prediction))

if __name__ == '__main__':
    main()
