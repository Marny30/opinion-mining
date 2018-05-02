#!/usr/bin/python3

import random

def getLabels():
    path = "./labels.csv"
    return [line.strip('\n') for line in open(path)]

def getDataset():
    # path = "./datasetlematise.csv"
    path = "./dataset.csv"
    return [line.strip('\n') for line in open(path)]

def main():
    labels = getLabels()
    dataset = getDataset()

    indices = list(range(len(dataset)))
    random.shuffle(indices)

    datasetshuffled = []
    labelshuffled = []
    for i in indices:
        datasetshuffled.append(dataset[i])
        labelshuffled.append(labels[i])

    with open("dataset_shuffled.csv", "w+") as datasetfile:
        datasetshuffled_str = "\n".join(datasetshuffled)
        datasetfile.write(datasetshuffled_str)
    with open("labels_shuffled.csv", "w+") as myfile:
        mystr = "\n".join(labelshuffled)
        myfile.write(mystr)
        
if __name__ == '__main__':
    main()
