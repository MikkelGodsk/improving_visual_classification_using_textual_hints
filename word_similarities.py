from glove_interface import glove_similarities, get_glove
from wordnet_interface import wn_similarities, get_wordnet
from spacy_interface import spacy_similarities, get_spacy

""" 
    Combines wordnet, glove etc. in order to get a unified way of specifying the arguments.
    word similarities should be a function of the type:  (str, str) -> float32
    The strings are synset names.
    
    To add word similarities:
        1) Add a file [package]_interface.py in which you define a dictionary of the similarities and possibly an
           init function. The dictionary should map from similarity name (without package name prefix) to a
           similarity function returning a float.
        2) Write a for-loop below to merge the dictionaries. Remember to prefix the key with the package name.
        3) In the init function below, add an elif-clause to initialize the interface (call the init function from 
           step 1).
    
    This would have been much nicer if we could just implement as classes with static name-variables. Then the classes
    could be added to a list and looped through, until the first prefix was found, from which we could instantiate an 
    object representing the similarities of the package.     
"""

word_similarities = {}
for k, v in wn_similarities.items():
    word_similarities['wordnet_' + k] = v

for k, v in glove_similarities.items():
    word_similarities['glove.6B.300d_' + k] = v
    word_similarities['glove.840B.300d_' + k] = v

for k, v in spacy_similarities.items():
    word_similarities['spacy_' + k] = v


def init(wd_sim_name):
    if "wordnet" in wd_sim_name.lower():
        get_wordnet("")
    elif "glove.6b.300d" in wd_sim_name.lower():
        get_glove(version='6B.300d')
    elif "glove.840b.300d" in wd_sim_name.lower():
        get_glove(version='840B.300d')
    elif "spacy" in wd_sim_name.lower():
        get_spacy("")