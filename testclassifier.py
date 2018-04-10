#!/usr/bin/python3

from classifierwrapper import ClassifierWrapper
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import confusion_matrix
import time

import utils

class ClassifierTester:
    def __init__(self, label, classifier):
        self.label = label
        self.clf_metrics = {}
        self.clf_metrics['label'] = label
        for dataset_type in utils.FORMATTED_DATA:
            # raw, lemma, without stopwords, ....
            self.clf_metrics[dataset_type] = {}
        self.clf = classifier
        
    def compute_metrics(self):
        for dataset_type in utils.FORMATTED_DATA:
            training_start_time = time.time()
            clfwrapper = ClassifierWrapper(
                utils.FORMATTED_DATA_TRAIN[dataset_type],
                utils.LABELS_TRAIN, self.clf)
            self.clf_metrics[dataset_type]['training_time'] = time.time() - training_start_time
            self._compute_metrics(dataset_type, clfwrapper)
        return self.clf_metrics

    def _compute_metrics(self, dataset_type, clfwrapper):
        analysis_start_time = time.time()
        analysis_result = self.clf_metrics[dataset_type]
        prediction =  clfwrapper.predict(
            utils.FORMATTED_DATA_TEST[dataset_type])
        l_test = utils.LABELS_TEST
        analysis_result['accuracy_score'] = accuracy_score(l_test, prediction)
        analysis_result['recall_score'] = recall_score(l_test, prediction)
        analysis_result['f1_score'] = f1_score(l_test, prediction)
        tn, fp, fn, tp = confusion_matrix(l_test, prediction).ravel()
        analysis_result['true_negative'] = tn
        analysis_result['true_positive'] = tp
        analysis_result['false_negative'] = fn
        analysis_result['false_positive'] = fp
        # Temps d'analyse en seconde
        analysis_result['prediction_time'] = time.time() - analysis_start_time
        
        
        # from sklearn.metrics import classification_report
        # print(self.label, "dataset : ", dataset_type)
        # print(classification_report(ClassifierTester.labels_test, prediction))
        
        return analysis_result

    # todo : confusion matrix
    
def main():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
    labels = [
        "K Neighbors n=2",
        "K Neighbors n=4",
        "K Neighbors n=7",
        "Naive Bayes",
        "Decision Tree",
        "SVM"
    ]
    classifiers = [
        KNeighborsClassifier(n_neighbors=2),
        KNeighborsClassifier(n_neighbors=4),
        KNeighborsClassifier(n_neighbors=7),
        MultinomialNB(),
        DecisionTreeClassifier(),
        svm.SVC()
        ]
    testers = []
    for i in range(len(labels)):
        testers.append(ClassifierTester(labels[i], classifiers[i]))

    import pprint
    pp = pprint.PrettyPrinter()
    for tester in testers:
        tester.compute_metrics()
        pp.pprint(tester.clf_metrics)
        
if __name__ == '__main__':
    main()
