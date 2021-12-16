import wordnet_interface as wnet
import imagenet_interface as inet
from embedding_similarities import embedding_similarities


import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds
import numpy as np


def get_bert():
    # Source: https://www.tensorflow.org/text/tutorials/classify_text_with_bert
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3'
    tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3'
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text_input")
    preprocessing = hub.KerasLayer(tfhub_handle_preprocess, name="preprocessing", trainable=False)
    encoder = hub.KerasLayer(tfhub_handle_encoder, name="BERT_encoder", trainable=False)
    return tf.keras.Model(text_input,
                          encoder(preprocessing(text_input))['pooled_output'])
                          #tf.reduce_mean(encoder(preprocessing(text_input))['sequence_output'], axis=1))  # What kind of pooling??


if __name__ == '__main__':
    bert = get_bert()

    wd_sims = wnet.wn_similarities.values()
    em_sims = [embedding_similarities['cosine']]

    # Get wordnet descriptions
    synsets = inet.get_dataset_labels()
    descriptions = []
    for synset_id in synsets:
        synset = wnet.find_synset(synset_id)
        descriptions.append(synset.definition())
    descriptions = tf.convert_to_tensor(descriptions)

    # Compute embeddings
    embeddings = bert(descriptions)
    N_similarities = int(1000 * (1000+1) / 2)
    embedding_similarities = np.zeros((len(em_sims), N_similarities), dtype=np.float32)
    wordnet_similarities = np.zeros((len(wd_sims), N_similarities), dtype=np.float32)

    # Compute embedding similarities
    print(embeddings.shape)
    for k, sim in enumerate(em_sims):
        h = 0
        for i in range(len(synsets)):
            for j in range(i, len(synsets)):
                embedding_similarities[k][h] = (
                    sim(embeddings[i:i + 1],
                        embeddings[j:j + 1])
                )
                h += 1

    # Compute WordNet similarities
    for k, sim in enumerate(wd_sims):
        h = 0
        for i in range(len(synsets)):
            for j in range(i, len(synsets)):
                wordnet_similarities[k][h] = (
                    sim(synsets[i],
                        synsets[j])
                )
                h += 1
    np.save(
        '/work3/s184399/similarity_experiment_BERT_WordNet/embedding_similarities_pooled.npy',
        embedding_similarities
    )
    np.save(
        '/work3/s184399/similarity_experiment_BERT_WordNet/wordnet_similarities_pooled.npy',
        wordnet_similarities
    )
