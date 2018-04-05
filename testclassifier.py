#!/usr/bin/python3

from classifierwrapper import ClassifierWrapper
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

SHUFFLED_DATASET_PATH = "./Données/dataset_shuffled.csv"
SHUFFLED_LABELS_PATH = "./Données/labels_shuffled.csv"
SHUFFLED_LEMMA_DATASET_PATH = "./Données/lemmatized_dataset_shuffled.csv"

def getLabels(path):
    return [int(line.strip('\n')) for line in open(path)]

def getDataset(path):
    return [line.strip('\n') for line in open(path)]

class ClassifierTester:
    # variables statiques
    data = getDataset(SHUFFLED_DATASET_PATH)
    data_lemma = getDataset(SHUFFLED_LEMMA_DATASET_PATH)
    labels = getLabels(SHUFFLED_LABELS_PATH)
    n = 5000
    data_train = data[:n]       # les n premiers
    labels_train = labels[:n]
    data_test = data[n:]        # les n derniers
    data_lemma_test = data_lemma[n:]
    labels_test = labels[n:]
    
    def __init__(self, label, classifier):
        self.label = label
        self.clfwrapper = ClassifierWrapper(ClassifierTester.data_train, ClassifierTester.labels_train, classifier)

    def getPrecision(self):
        prediction = self.clfwrapper.predict(ClassifierTester.data_test)
        return accuracy_score(ClassifierTester.labels_test, prediction)

    def getPrecisionLemma(self):
        prediction = self.clfwrapper.predict(ClassifierTester.data_lemma_test)
        return accuracy_score(ClassifierTester.labels_test, prediction)
    
    def __str__(self):
        return (
            "{:20} {:15} {:15}".format(self.label, self.getPrecision(), self.getPrecisionLemma()))
    
def main():
    labels = [
        "K Neighbors n=2",
        "K Neighbors n=4",
        "K Neighbors n=7"
    ]
    classifiers = [
        KNeighborsClassifier(n_neighbors=2),
        KNeighborsClassifier(n_neighbors=4),
        KNeighborsClassifier(n_neighbors=7)
        ]
    testers = []
    print("{:20} {:15} {:15}".format("label", "precision_brut", "precision_lemma"))
    for i in range(len(labels)):
        testers.append(ClassifierTester(labels[i], classifiers[i]))

    for tester in testers:
        print(tester)
        
if __name__ == '__main__':
    main()
