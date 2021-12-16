import spacy
import re

from wordnet_interface import find_synset, splitting_chars

nlp = None


def get_spacy(version):
    global nlp  # !!
    nlp = spacy.load('en_core_web_lg')


def is_initialized():
    return nlp is not None


def spacy_similarity(x, y):
    # Get the label
    x = find_synset(x).name().split('.')[0]
    y = find_synset(y).name().split('.')[0]

    # Replace -,_ etc. with space
    x = re.sub(splitting_chars, ' ', x)
    y = re.sub(splitting_chars, ' ', y)

    return nlp(x).similarity(nlp(y))


spacy_similarities = {
    'similarity': spacy_similarity  # ? Which similarity is this?
}
