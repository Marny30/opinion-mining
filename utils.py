#!/usr/bin/python3

from sklearn.feature_extraction.text import TfidfVectorizer

def getLabels(path):
    return [int(line.strip('\n')) for line in open(path)]

def getDataset(path):
    return [line.strip('\n') for line in open(path)]

SHUFFLED_DATASET_PATH = "./Données/dataset_shuffled.csv"
SHUFFLED_LABELS_PATH = "./Données/labels_shuffled.csv"
SHUFFLED_LEMMA_DATASET_PATH = "./Données/lemmatized_dataset_shuffled.csv"
SHUFFLED_LEMMA_MORPHO_DATASET_PATH = "./Données/lemmatized_morpho_dataset_shuffled.csv"
SHUFFLED_WITHOUT_STOPWORDS_DATASET_PATH = "./Données/dataset_without_stopwords_shuffled.csv"
LABELS = getLabels(SHUFFLED_LABELS_PATH)

# Les données bruts ne sont pas censés apporter d'information
# intéressante pour le reste du projet (contrairement aux données
# formattées), d'où le prefixe underscore
_DATA = {}
_DATA['raw'] = getDataset(SHUFFLED_DATASET_PATH)
_DATA['without-stopwords'] = getDataset(SHUFFLED_WITHOUT_STOPWORDS_DATASET_PATH)
_DATA['lemma-morpho'] = getDataset(SHUFFLED_LEMMA_MORPHO_DATASET_PATH)
_DATA['lemma'] = getDataset(SHUFFLED_LEMMA_DATASET_PATH)

FORMATTED_DATA_TRAIN = {}
FORMATTED_DATA_TEST = {}
LABELS_TRAIN = []
LABELS_TEST = []


class DataAdaptater():
    """Classe permettant de transformer des données bruts en données
    tf-idf pour permettre l'analyse par les classifieurs

    """
    vectorizer = TfidfVectorizer()
    # tfidfVectorizer est équivalent à un CountVectorizer suivi d'un tfidf_transformer
    # ref : doc ( http://scikit-learn.org/stable/modules/feature_extraction.html#common-vectorizer-usage )

    @staticmethod
    def adapt_train(dataset):
        return DataAdaptater.vectorizer.fit_transform(dataset)

    @staticmethod
    def adapt_test(dataset):
        return DataAdaptater.vectorizer.transform(dataset)

def define_train_data(dataset, dataset_type):
    """ transforme un dataset (non formatté) et le défini comme dataset d'entrainement """
    FORMATTED_DATA_TRAIN[dataset_type] = DataAdaptater.adapt_train(dataset)
        
def define_test_data(dataset, dataset_type):
    """ transforme un dataset (non formatté) et le défini comme dataset de test """
    FORMATTED_DATA_TEST[dataset_type] = DataAdaptater.adapt_test(dataset)

def define_train_labels(labels):
    global LABELS_TRAIN
    LABELS_TRAIN = labels

def define_test_labels(labels):
    global LABELS_TEST
    LABELS_TEST = labels

def prediction_config():
    """Définie la configuration pour la prédiction, où les labels
    résultants sont inconnus. Le dataset d'entrainement n'est ici pas
    le même que le dataset de test.
    """
    define_train_labels(LABELS)
    
    path_test = {}
    path_test['raw'] = "./Données/test_data.csv"
    path_test['without-stopwords'] = "./Données/test_data_without_stopwords.csv"
    path_test['lemma'] = "./Données/test_data_lemmatized.csv"
    path_test['lemma-morpho'] = "./Données/test_data_lemmamorpho.csv"
    for key in _DATA:
        text_train = _DATA[key]
        define_train_data(text_train, key)
        # Jeu de donné inchangé, pour chaque entraînement distinct
        define_test_data(getDataset(path_test[key]), key)
        
def default_config(partie_entrainement=0.66):
    """Définie la configuration par défaut pour les sous-programmes. La
    configuration par défault est la suivante : 
    - Un seul fichier est lu.
    - les n premières données sont utilisées pour l'entrainement, les n 
    suivantes sont utilisées pour le test

    partie_entrainement est défini par défaut de manière à respecter
    la proportion 2/3 entrainement, 1/3 test.
    """
    n = int(partie_entrainement * len(_DATA['raw']))
    define_train_labels(LABELS[:n])
    define_test_labels(LABELS[n:])
    for key in _DATA:
        text_train = _DATA[key][:n]
        text_test = _DATA[key][n:]
        define_train_data(text_train, key)
        define_test_data(text_test, key)
default_config()
