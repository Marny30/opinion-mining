#!/usr/bin/python3

from classifierwrapper import ClassifierWrapper
from sklearn.metrics import accuracy_score, recall_score, f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import confusion_matrix
import time
from sklearn.model_selection import cross_val_score
from sklearn import svm
import utils
import scikitplot as skplt
import matplotlib.pyplot as plt
from sklearn import metrics
import numpy as np


class ClassifierTester:
    def __init__(self, label, classifier):
        self.label = label
        self.clf_metrics = {}
        self.clf_metrics['label'] = label
        for dataset_type in utils._DATA:
            # raw, lemma, without stopwords, ....
            self.clf_metrics[dataset_type] = {}
        self.clfwrapper =  ClassifierWrapper(classifier)
        self.classifier = classifier
        
    def compute_metrics(self):
        for dataset_type in utils._DATA:
            training_start_time = time.time()
            self.clfwrapper.train(utils.FORMATTED_DATA_TRAIN[dataset_type], utils.LABELS_TRAIN)
            self.clf_metrics[dataset_type]['training_time'] = time.time() - training_start_time
            self._compute_metrics(dataset_type, self.clfwrapper)
        return self.clf_metrics

    def _compute_metrics(self, dataset_type, clfwrapper):
        analysis_start_time = time.time()
        analysis_result = self.clf_metrics[dataset_type]
        prediction =  clfwrapper.predict(
            utils.FORMATTED_DATA_TEST[dataset_type])
        l_test = utils.LABELS_TEST
        cross_validation = cross_val_score(self.classifier, utils.FORMATTED_DATA_TRAIN[dataset_type], utils.LABELS_TRAIN, cv=10)
        analysis_result['cross_validation'] = cross_validation.mean()
        analysis_result['marge_cross_validation (+/-)'] = cross_validation.std() * 2
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


        ###### CODE D'AFFICHAGE DE ROC CURVE ##########
        probs = self.classifier.predict_proba(utils.FORMATTED_DATA_TEST[dataset_type])
        preds = probs[:,1]
        fpr, tpr, threshold = metrics.roc_curve(utils.LABELS_TEST, preds)
        roc_auc = metrics.auc(fpr, tpr)
        plt.title('ROC CURVE - ' + self.label)
        plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
        plt.legend(loc = 'lower right')
        plt.plot([0, 1], [0, 1],'r--')
        plt.xlim([0, 1])
        plt.ylim([0, 1])
        plt.ylabel('True Positive Rate')
        plt.xlabel('False Positive Rate')
        plt.show()
        ################################################

        #y_true = np.asarray(utils.LABELS_TEST)
        #y_probas = np.asarray(prediction)
        #fpr, tpr, thresholds = metrics.roc_curve(y_true, y_probas, pos_label=5)

        # Print ROC curve
        #plt.plot(fpr,tpr)
        #plt.show() 

        # Print AUC
        #auc = np.trapz(tpr,fpr)
        #print('AUC:', auc)



        # from sklearn.metrics import classification_report
        # print(self.label, "dataset : ", dataset_type)
        # print(classification_report(ClassifierTester.labels_test, prediction))        
        return analysis_result

    # TODO
    # def get_comment_giving_wrong_predictions(self, dataset_type, n=0):
    #     formatted_data = utils.FORMATTED_DATA_TEST[dataset_type]
    #     raw_data = utils._DATA_TEST[dataset_type]
    #     res = {'false_positive' : [], 'false_negative' : []}
    #     t_labels = utils.LABELS_TEST
    #     if n == 0:
    #         n = len(utils.LABELS_TEST)  # nombre de donnée de test
    #     # relance une prédiction..!
    #     prediction = self.clfwrapper.predict(formatted_data)
        
    #     for i in range(len(t_labels)):
    #         if utils.LABELS_TEST[i] != prediction[i]:
    #             if utils.LABELS_TEST[i] == 1:
    #                 res['false_negative'] += [raw_data[i]]
    #             elif utils.LABELS_TEST[i] == -1:
    #                 res['false_positive'] += [raw_data[i]]
    #             else:
    #                 raise ValueError('Formattage utils.LABELS_TEST[i]')

        # on ne retient que les n premiers avis
        # res['false_negative'] = res['false_negative'][:n]
        # res['false_positive'] = res['false_positive'][:n]
        # return res

def main():
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.linear_model import SGDClassifier
    
    
    # TODO : jouer variation C
    classifiers = [
        ["K Neighbors n=1", KNeighborsClassifier(n_neighbors=1)],#
        ["K Neighbors n=5", KNeighborsClassifier(n_neighbors=5)],
        ["K Neighbors n=10", KNeighborsClassifier(n_neighbors=10)],
        ["Naive Bayes", MultinomialNB()],
        ["Decision Tree", DecisionTreeClassifier()],
        ["SVM linéaire", svm.SVC(kernel='linear')],
        ["SVM polynominal degré 3", svm.SVC(kernel='poly', degree=3)],
        ["SVM rbf, gamma=0.7", svm.SVC(kernel='rbf', gamma=0.7)],
        ["SVM rbf, gamma=0.9", svm.SVC(kernel='rbf', gamma=0.9)],
        ["SVM rbf, gamma=1/5000", svm.SVC(kernel='rbf')],
        ["SGDClassifier", SGDClassifier()]
    ]
    
    testers = []
    for couple_label_clf in classifiers:
        testers.append(ClassifierTester(couple_label_clf[0], couple_label_clf[1]))

    import pprint
    pp = pprint.PrettyPrinter()
    # for tester in testers:
        # tester.compute_metrics()
        # print(tester.label)
        # for dataset_type in utils._DATA:
        # dataset_type = 'raw'
        # print("jeu de donnée :" , dataset_type)
        # tester.clfwrapper.train(utils.FORMATTED_DATA_TRAIN[dataset_type], utils.LABELS_TRAIN)
        # res = tester.get_comment_giving_wrong_predictions(dataset_type)
        # for key in res:
        #     print(key, res[key])
            # pp.pprint(res)
            # print(res)
            # print()
   
    for tester in testers:
        tester.compute_metrics()    
        pp.pprint(tester.clf_metrics)


if __name__ == '__main__':
    main()
