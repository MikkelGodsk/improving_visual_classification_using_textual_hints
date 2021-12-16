import json
import tensorflow as tf
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

from models import get_bert
import wordnet_interface as wnet
import imagenet_interface as inet

class BertWrapper(tf.keras.Model):
    def __init__(self,
                 bert_final_op='pooled_output',
                 dense_layer_units=None,
                 kernel_regularizer=None,
                 n_classes=1000,
                 *args, **kwargs):
        super(BertWrapper, self).__init__(*args, **kwargs)

        # BERT part of the network
        self.bert = get_bert(final_op=bert_final_op)

        # Dense layers before concatenation
        self.dense_layers = tf.keras.Sequential([])  # tf.keras.layers.Dropout(rate=0.3)])  # Can it be annealed?
        if dense_layer_units is not None:
            for units in dense_layer_units:
                self.dense_layers.add(tf.keras.layers.Dense(units=units,
                                                            trainable=True,
                                                            kernel_regularizer=kernel_regularizer))
                self.dense_layers.add(tf.keras.layers.BatchNormalization())
                self.dense_layers.add(tf.keras.layers.ReLU())
                self.dense_layers.add(tf.keras.layers.Dropout(rate=0.2))
        self.dense_layers.add(tf.keras.layers.Dense(units=n_classes,
                                                    trainable=True,
                                                    kernel_regularizer=kernel_regularizer))

    def call(self, x, training=False):
        txt = x  # ['description']
        txt_embedding = self.bert(txt)
        txt_output = self.dense_layers(txt_embedding, training=training)

        return tf.nn.softmax(txt_output)


def mask_words_in_string(s, p=None, seed=None):
    """
        An auxiliary function for mask_words.
        Removes int((1-p)*len(s)) words from the string s.
        Order is preserved, and amount of words is rounded up (so it will only be zero if p == 0)

        If decreasing_p is True, then p becomes obsolete.

        Args:
            p: tf.Tensor object - The proportion of words to mask.
            s: string object (tf.String?) - The string to mask
            decreasing_p: boolean - Whether we should decrease p over time. Ideally, this could be a strategy
                                    implemented in another class (a base class and a bunch of subclasses)
    """
    # If p is not given (i.e. a tf.constant with value -1), then we select it randomly.
    p = tf.cond(tf.math.greater_equal(p, tf.constant(0.0)),  # is p>=0
                true_fn=lambda: p,
                false_fn=lambda: tf.random.uniform(shape=[], minval=0.0, maxval=1.0, seed=seed),
                name='Select_p'
                )
    # Split string and compute amount of words to keep
    s_split = tf.strings.split(s)
    # s_split = tf.strings.split(s, sep='-')
    n_words = tf.cast(tf.size(s_split), dtype=tf.float32)
    n_kept_words = tf.cast(tf.math.ceil(p * n_words), dtype=tf.int32)

    # Compute indices of randomly dropped words.
    idx = tf.range(0, n_words, dtype=tf.int32)
    idx = tf.random.shuffle(idx, seed=seed)
    idx = idx[:n_kept_words]
    idx = tf.sort(idx)

    out = tf.gather(s_split, idx)
    out = tf.strings.reduce_join(out, separator=' ')
    # This is a sentence  ->   4 words -> 0 1 2 3  ->  1 0 3 2 -> 1 0 3 -> 0 1 3 -> This is sentence = syntactical nonsense, may hurt BERT
    # Find the string corresponding to the UNKNOWN token and replace with that. Does it make a difference?

    return out


def get_wordnet_descriptions():
    """
        Goes through the wordnet taxonomy and obtains the definitions for the classes in imagenet.
    """
    descriptions = {}
    for i, label in enumerate(inet.get_dataset_labels()):
        desc = wnet.find_synset(label).definition()
        descriptions[label] = desc
    return descriptions


def load_and_preprocess_dataset():
    wiki_descriptions = 'wiki_descriptions.json'
    with open(wiki_descriptions, 'r') as f_obj:
        descriptions = json.load(f_obj)

    wn_desc = get_wordnet_descriptions()

    X_wiki = []
    y_wiki = []
    X_wn = []
    y_wn = []

    labels = []

    i = 0
    for k, vs in descriptions.items():
        ds = vs.replace('\n', ' ').split('. ')
        if len(ds) > 1:
            for v in ds:
                X_wiki.append(v)
                y_wiki.append(i)
                labels.append(k)
            X_wn.append(wn_desc[k])  # Find synset and add description
            y_wn.append(i)
            i += 1

    X_wiki = np.array(X_wiki)
    y_wiki = np.array(y_wiki)
    X_wn = np.array(X_wn)
    y_wn = np.array(y_wn)
    n_classes = i

    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2)
    for train_ix, test_ix in sss.split(X_wiki, y_wiki):
        ixs = np.hstack((train_ix, test_ix))

    X_wiki = tf.convert_to_tensor(X_wiki[ixs])
    y_wiki = tf.convert_to_tensor(y_wiki[ixs])
    X_wn = tf.convert_to_tensor(X_wn)
    y_wn = tf.convert_to_tensor(y_wn)
    #data = tf.data.Dataset.from_tensors((X_wiki, y_wiki))

    return n_classes, X_wiki, y_wiki, X_wn, y_wn


if __name__ == '__main__':
    # Train on wikipedia, test on wordnet?

    n_classes, X_wiki, y_wiki, X_wn, y_wn = load_and_preprocess_dataset()
    #train = train.batch(20).prefetch(tf.data.AUTOTUNE)
    #val = val.batch(20).prefetch(tf.data.AUTOTUNE)
    model = BertWrapper(
        bert_final_op='avg sequence',
        n_classes=n_classes,
        #kernel_regularizer=tf.keras.regularizers.l2()
    )
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5, name='sparse_top_5_categorical_accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=1, name='sparse_top_1_categorical_accuracy')
        ]
    )
    model2 = BertWrapper(
        bert_final_op='pooled_output',
        n_classes=n_classes
    )
    model2.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(
            k=5, name='sparse_top_5_categorical_accuracy'),
            tf.keras.metrics.SparseTopKCategoricalAccuracy(
                k=1, name='sparse_top_1_categorical_accuracy')
        ]
    )

    with tf.device('/device:GPU:0'):
        print("Avg seq:")
        model.fit(x=X_wiki, y=y_wiki, validation_split=0.2, epochs=5, verbose=1)
        model.evaluate(x=X_wn, y=y_wn)
        model.evaluate(x=tf.map_fn(fn=lambda x: mask_words_in_string(s=x,
                                                                     p=tf.constant(0.7)),
                                                                     elems=X_wn),
                       y=y_wn)
        print("\nPooled output:")
        model2.fit(x=X_wiki, y=y_wiki, validation_split=0.2, epochs=5, verbose=1)
        model2.evaluate(x=X_wn, y=y_wn)
        model2.evaluate(x=tf.map_fn(fn=lambda x: mask_words_in_string(s=x,
                                                                     p=tf.constant(0.7)),
                                   elems=X_wn),
                       y=y_wn)
