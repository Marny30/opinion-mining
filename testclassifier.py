#!/usr/bin/python3

from classifierwrapper import ClassifierWrapper
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import precision_score
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
        analysis_result['precision_score'] = precision_score(l_test, prediction)
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

    # TODO
    # def get_comment_giving_wrong_predictions(self, dataset_type):
    #     res = []
    #     t_labels = utils.LABELS_TEST
    #     data = utils.FORMATTED_DATA_TEST[dataset_type]
    #     prediction = self.clf.predict(data)
        
    #     for i in range(len(t_labels)):
    #         if utils.LABELS_TEST[i] != prediction[i]:
    #             res += data[i]
    
def main():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn import svm
    labels = [
        # "K Neighbors n=1",
        # "K Neighbors n=5",
        # "K Neighbors n=10",
        # "Naive Bayes",
        # "Decision Tree",
        "SVM"
    ]

    classifiers = [
        ["K Neighbors n=1", KNeighborsClassifier(n_neighbors=1)],
        ["K Neighbors n=5", KNeighborsClassifier(n_neighbors=5)],
        ["K Neighbors n=10", KNeighborsClassifier(n_neighbors=10)],
        ["Naive Bayes", MultinomialNB()],
        ["Decision Tree", DecisionTreeClassifier()],
        ["SVM linéaire", svm.SVC(kernel='linear')],
        ["SVM polynominal degré 3", svm.SVC(kernel='poly', degree=3)],
        ["SVM rbf, gamma=0.7", svm.SVC(kernel='rbf', gamma=0.7)],
        ["SVM rbf, gamma=0.9", svm.SVC(kernel='rbf', gamma=0.9)],
        ["SVM rbf, gamma=1/5000", svm.SVC(kernel='rbf')]
    ]
    
    testers = []
    for couple_label_clf in classifiers:
        testers.append(ClassifierTester(couple_label_clf[0], couple_label_clf[1]))

    import pprint
    pp = pprint.PrettyPrinter()
    for tester in testers:
        tester.compute_metrics()
        pp.pprint(tester.clf_metrics)
        
if __name__ == '__main__':
    main()
