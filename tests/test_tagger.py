#!/usr/bin/python3

# ref : https://docs.pytest.org/en/latest/
# ref : https://docs.pytest.org/en/latest/pythonpath.html

# pip3 install --user pytest
# pour lancer : deux solutions (depuis racine du dossier)
#   - py.test-3 (lancer tous les tests)
#   - python3 -m pytest ./tests/test_tagger.py "-s"

import pretraitement
import treetaggerwrapper

def test_poncutation_point():
    '''Test s'il n'y a pas de problème de taggage (à cause de la
ponctuation notamment)'''
    data = "this is a tests.that could bug"  # Générait un NotTag(tests.that)
    formatted = pretraitement.formattage(data)
    tags = pretraitement.extract_tags(formatted)
    assert all( (type(tag) is treetaggerwrapper.Tag) for tag in tags)

def test_url():
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="news:" />')
    # PROBLEM:  NotTag(what='<repurl text="http://imdb." />')
    # PROBLEM:  NotTag(what='<repurl text="file:" />')
    # PROBLEM:  NotTag(what='<repurl text="http://skeptico." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://en." />')
    # PROBLEM:  NotTag(what='<repurl text="http://memory-alpha." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://ferdinandvongalitzien." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<re1purl text="news:" />')
    # PROBLEM:  NotTag(what='<repurl text="http://friderwaves." />')
    # PROBLEM:  NotTag(what='<repurl text="http://store." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    # PROBLEM:  NotTag(what='<repurl text="http://www." />')
    data = 'this is a url <repurl text="http://memory-alpha." />.'  # Générait un NotTag
    formatted = pretraitement.formattage(data)
    tags = pretraitement.extract_tags(formatted)
    assert all( (type(tag) is treetaggerwrapper.Tag) for tag in tags)

def test_whole_dataset():
    # with open("./Données/dataset.csv", "r") as myfile:
    with open("./Données/dataset.csv", "r") as myfile:
        data = myfile.read() # Générait un NotTag(tests.that)
    formatted = pretraitement.formattage(data)
    tags = pretraitement.extract_tags(formatted)
    for t in tags:
        if (type(t) != treetaggerwrapper.Tag): print("PROBLEM: ", t)
    assert all( (type(tag) is treetaggerwrapper.Tag) for tag in tags)
