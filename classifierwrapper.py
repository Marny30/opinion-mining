#!/usr/bin/python3

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

#ref : http://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        
class ClassifierWrapper:
    """ Classifieur paramétrable selon le type de classifieur souhaité
    """
    def __init__(self, data_train_tfidf, labels, clf):
        # TF-idf : Term Frequency times inverse document frequency
        self.clf = clf
        # print("entrainement sur " + len(data_train_tfidf) + " données ")
        self.clf.fit(data_train_tfidf, labels)
        
    def predict(self, data_test_tfdif):
        return self.clf.predict(data_test_tfdif)

    def visualization(self): # TODO
        pass

# note pour plus tard : pour n=3, dataset non lemmatisé plus précis, mais si n augmente cet écart s'amoindri
def main():
    import utils

    classifier = KNeighborsClassifier(n_neighbors=4)
    # autre exemple
    # classifier = MultinomialNB()
    # print(utils.FORMATTED_DATA_TRAIN['raw'])
    k_neighbors_classifier = ClassifierWrapper(utils.FORMATTED_DATA_TRAIN['raw'], utils.LABELS_TRAIN, classifier)
    prediction = k_neighbors_classifier.predict(utils.FORMATTED_DATA_TEST['raw'])
    print("Précision pour kneighbors (n=3) : ", accuracy_score(utils.LABELS_TEST, prediction))

if __name__ == '__main__':
    main()
