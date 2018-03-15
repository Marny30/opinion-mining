#!/usr/bin/python3


import treetaggerwrapper


def extract_tags(sentence):
    '''Extraction des tags (Mot, classe grammaticale, lemme) depuis le
    treetaggerwrapper
    '''
    import os
    ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    TREETAGGER_DIR = ROOT_DIR+"/dep/treetagger"
    tagger = treetaggerwrapper.TreeTagger(TAGLANG='en',
                                          TAGDIR=TREETAGGER_DIR)
    tagged_text = tagger.tag_text(sentence)
    tags = treetaggerwrapper.make_tags(tagged_text)
    return tags

def formattage(sentence):
    ''' Formattage de la phrase : suppression de toutes les lettres non alphanumériques '''
    res = sentence.lower()
    # TODO : virer "!" ?
    # filtre de tous les éléments non alphanumériques
    # On garde l'apostrophe pour les constructions du style "He has/He is => He's" ou pour le possessif
    # res = "".join(
    #     [
    #     letter for letter in sentence if
    #     (letter.isalpha() or letter.isspace() or letter=="'")
    # ]
    # )
    return res

def selection(tags):
    ''' fonction de selection des tags '''
    interestingPOS=["VBG","NN","NNS","VBR","VBZ","JJS","RBS","VB","VBD","JJ","JJR","JJS","VBN"]
    res=[]
    for tag in tags:
        if (tag.pos  in  interestingPOS):
            res+=[{'pos': tag.pos, 'lemma':tag.lemma}]
    return res

def pretraitement(sentence):
    formatted = formattage(sentence)
    tags = extract_tags(formatted)
    return selection(tags)
def Entree():
    import csv
    Text=[]    
    taille=int(input("Entrez le nombre de ligne que vous voulez (1 à 10000) : "))
    while(taille<1 or taille>10000):
        taille=int(input("Entrez le nombre de ligne que vous voulez (1 à 4000) : "))
    with open("../Données/dataset.csv") as csvfile:
    #Text=csvfile.read().split(' \n')
        for i in range(taille):
            Text+=[csvfile.readline()]
    for i in range(taille):
        formatted = formattage(Text[i])
        print("Phrase formattée :")
        print(formatted)
        
        tags = extract_tags(formatted)
        print(tags)
        for tag in tags:
            print(tag)
        select=selection(tags)
        print(select)

def main():
    import sys, csv
    if(len(sys.argv)==1):
        Entree()
    else:
        my_sentence = sys.argv[1]
        print("Input : ")
        print(my_sentence)
        formatted = formattage(my_sentence)
        # formatted = my_sentence
        print("Phrase formattée :")
        print(formatted)
        tags = extract_tags(formatted)
        print(tags)
        for tag in tags:
            print(tag)
        select=selection(tags)
        print(select)

if __name__ == "__main__":
    main()
