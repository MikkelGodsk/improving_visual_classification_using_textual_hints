import nltk
from nltk.corpus import wordnet as wn

splitting_chars = "_| |-|'"  # | as separator
initialized = False

def find_synset(wnid):
    pos, offset = wnid[0], int(wnid[1:])
    return wn.synset_from_pos_and_offset(pos, offset)

def get_wordnet(version):
    global initialized
    nltk.download('wordnet')
    initialized = True

def is_initialized():
    return initialized

wn_similarities = {
    'path': lambda x, y: find_synset(x).path_similarity(find_synset(y)),
    'lch': lambda x, y: find_synset(x).lch_similarity(find_synset(y)),
    'wup': lambda x, y: find_synset(x).wup_similarity(find_synset(y)),
}