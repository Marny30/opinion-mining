#!/usr/bin/python3

import treetaggerwrapper

import os
_ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
_TREETAGGER_DIR = _ROOT_DIR+"/dep/treetagger"
_TAGGER = treetaggerwrapper.TreeTagger(TAGLANG='en',
                                       TAGDIR=_TREETAGGER_DIR)

def extract_tags(sentence):
    '''Méthode d'extraction des tags (voir _extract_tags) avec exclusion
    des éléments n'ayant pas pu être taggués
    '''
    tags = _extract_tags(sentence)
    res = list(filter(lambda x: isinstance(x, treetaggerwrapper.Tag), tags))
    return res

def _extract_tags(sentence):
    ''' Méthode interne d'extraction des tags (Mot, classe grammaticale, lemme) depuis le
    treetaggerwrapper
    '''
    tagged_text = _TAGGER.tag_text(sentence)
    tags = treetaggerwrapper.make_tags(tagged_text)
    return tags

def formattage(sentence):
    ''' Formattage de la phrase : suppression de toutes les lettres non alphanumériques '''
    res = sentence.lower()
    # Remplacer "." par ". "
    res = res.replace(".", ". ")
    # On garde l'apostrophe pour les constructions du style "He has/He
    # is => He's" ou pour le possessif
    ponctuation_to_keep = [',', '.', ';', '"', "'", ' ']
    res = "".join([
        letter for letter in sentence if
        (letter.isalpha() or letter in ponctuation_to_keep)])
    return res

def pos_selection(tags):
    ''' fonction de selection des tags
    liste des classes grammaticales :
    http://www.cis.uni-muenchen.de/~schmid/tools/TreeTagger/data/Penn-Treebank-Tagset.pdf'''
    # interestingPOS=["VBG","NN","NNS","VBR","VBZ","JJS","RBS","VB","VBD","JJ","JJR","JJS","VBN"]
    # TODO : negation? RB
    interestingPOS = ["NN","NNS","JJS", "RBS","JJ","JJR","JJS"]
    res = []
    for tag in tags:
        if (tag.pos in interestingPOS):
            res += [{'pos': tag.pos, 'lemma':tag.lemma}]
    return res


def selection_lemma(selected):
    return " ".join([x['lemma'] for x in selected])

def pretraitement(sentence, lemmatize=True, selection=True):
    formatted = formattage(sentence)
    tags = extract_tags(formatted)
    if selection:
        tags = pos_selection(tags)
    else:
        return " ".join([x.lemma for x in tags])
    
    if lemmatize:
        return selection_lemma(tags)
    else:
        return selection


def Entree():
    import csv
    Text=[]    
    taille=int(input("Entrez le nombre de ligne que vous voulez (1 à 10000) : "))
    while(taille<1 or taille>10000):
        taille=int(input("Entrez le nombre de ligne que vous voulez (1 à 4000) : "))
    with open("./Données/dataset.csv") as csvfile:
        for i in range(taille):
            Text+=[csvfile.readline()]
    return Text

def main():
    ''' Étape 1 '''
    import sys
    out_path = "./Données/lemmatized_dataset_shuffled.csv"
    in_path = "./Données/dataset_shuffled.csv"
    data = [line.strip('\n') for line in open(in_path)]
    chunksize = 10
    indice = 0

    # obtention du nb de lignes déjà prétraitées ##############################
    try: indice = len([line.strip('\n') for line in open(out_path, "r")])
    except: indice= 0
    
    if indice > len(data):
        print("[!] len(données_traitées) > len(dataset_donné)\n" +
              "    Effacement du fichier de sortie.", file=sys.stderr)
        myfile = open(out_path, "w+")
        myfile.close()
        indice = 0
    elif indice == len(data):
        print("le dataset est déjà prétraité! fermeture..", file=sys.stderr)
        exit()


    # prétraitement des données  ##############################################
    print("pretraitement à partir de i =", indice, file=sys.stderr)
    while indice != len(data):
        # generation de chunksize avis lemmatisé ##################################
        newdata = []
        next_i = min(indice + chunksize, len(data))
        for i in range(indice, next_i):
            newdata += [pretraitement(data[i], lemmatize=True, selection=False)]
            indice += 1
        print(indice, "/", len(data), file=sys.stderr)

        # écriture ################################################################
        print("Écriture..", end='', file=sys.stderr)
        sys.stderr.flush()
        with open(out_path, "a+") as lemma_file:
            for line in newdata:
                lemma_file.write(line+'\n')
        print("\t ok", file=sys.stderr)
        sys.stderr.flush()

if __name__ == "__main__":
    main()
