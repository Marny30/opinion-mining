#!/usr/bin/python3
from sklearn.neighbors import KNeighborsClassifier

def save_in_file(path, labels):
    with open(path, "w") as myfile:
        for label in labels:
            myfile.write(str(label)+'\n')
    
def main():
    import utils
    import classifierwrapper
    utils.prediction_config("./Données/test_data.csv")

    output_path_prefix = "./Données/test_labels-trained-with-"
    output_path_suffix = ".csv"

    # TODO : choix classifieur
    classifier = KNeighborsClassifier(n_neighbors=4)

    clfwrapper = classifierwrapper.ClassifierWrapper(classifier)
    for datatype in utils.FORMATTED_DATA_TRAIN:
        clfwrapper.train(utils.FORMATTED_DATA_TRAIN[datatype], utils.LABELS_TRAIN)
        data = utils.FORMATTED_DATA_TEST[datatype]
        # TODO : prétraitement donnée de test?
        # les résultats pour lemma-morpho sont abhérrent sinon
        prediction = clfwrapper.predict(data)
        filepath = output_path_prefix + datatype + output_path_suffix
        save_in_file(filepath, prediction)
        print("prediction done for classifier trained with", datatype,
              "dataset. output :", filepath)

if __name__ == '__main__':
    main()
