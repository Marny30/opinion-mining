#!/usr/bin/python3

# from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer

DATASET_PATH = "./Données/dataset.csv"
LABELS_PATH = "./Données/labels.csv"

SHUFFLED_DATASET_PATH = "./Données/dataset_shuffled.csv"
SHUFFLED_LABELS_PATH = "./Données/labels_shuffled.csv"
SHUFFLED_LEMMA_DATASET_PATH = "./Données/lemmatized_dataset_shuffled.csv"


class DataAdaptater():
    count_vect = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    def __init__(self):
        pass
    def adapt(self, dataset):
        data_counts = DataAdaptater.count_vect.fit_transform(dataset)
        # TF-idf : Term Frequency times inverse document frequency
        data_tfidf = DataAdaptater.tfidf_transformer.fit_transform(data_counts)
        return data_tfidf

    
def getLabels(path):
    return [int(line.strip('\n')) for line in open(path)]

def getDataset(path):
    # path = "./datasetlematise.csv"
    return [line.strip('\n') for line in open(path)]


LABELS = {}
LABELS = getLabels(SHUFFLED_LABELS_PATH)

n = 5000                        # todo : 2/3 1/3
LABELS_TRAIN = LABELS[:n]
LABELS_TEST = LABELS[n:]

# Les données bruts ne sont pas censés apporter d'information
# intéressante pour le reste du projet (contrairement aux données
# formattées), d'où le prefixe underscore
_DATA = {}
_DATA['raw'] = getDataset(SHUFFLED_DATASET_PATH)
_DATA['lemma'] = getDataset(SHUFFLED_LEMMA_DATASET_PATH)
_DATA_TRAIN = {}
_DATA_TRAIN['raw'] = _DATA['raw'][:n] # les n premiers
_DATA_TRAIN['lemma'] = _DATA['lemma'][:n]
_DATA_TEST = {}
_DATA_TEST['raw'] = _DATA['raw'][n:]        # les n derniers
_DATA_TEST['lemma'] = _DATA['lemma'][n:]


FORMATTED_DATA = {}
FORMATTED_DATA_TRAIN = {}
FORMATTED_DATA_TEST = {}
_da  = DataAdaptater()
for key in _DATA_TRAIN:
    FORMATTED_DATA[key] = _da.adapt(_DATA[key])
    FORMATTED_DATA_TRAIN[key] = FORMATTED_DATA[key][:n]
    FORMATTED_DATA_TEST[key] = FORMATTED_DATA[key][n:]
    
