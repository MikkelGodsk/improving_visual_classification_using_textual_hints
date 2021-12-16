import tensorflow as tf
import tensorflow_datasets as tfds
import json

import wordnet_interface as wnet
import imagenet_interface as inet
from networks import network_architectures


def pick_random_sentence(sentences, seed=None):
    sentence = tf.cond(tf.math.greater(tf.size(sentences), tf.constant(0)),
                       true_fn=lambda: sentences[tf.random.uniform(
                           shape=(),
                           maxval=tf.size(sentences),
                           dtype=tf.int32,
                           seed=seed
                       )],
                       false_fn=lambda: tf.constant('')
                       )
    return sentence


def prepare_sample(sample, descriptions, seed=None):
    """
        Prepares a sample (an image) from the dataset by adding a description and the label.
    """
    with tf.device('/cpu:0'):
        img = network_architectures['resnet50']['preprocess_input'](tf.image.resize(sample['image'], (224, 224)))
        # img = tf.expand_dims(img, axis=0)

        return {'image': img, 'description': pick_random_sentence(descriptions[sample['label']], seed=seed)}, sample['label']


def preprocess_dataset(dataset, descriptions, seed=None):
    return dataset.map(lambda x: prepare_sample(x, descriptions, seed=seed))


def make_batched_and_prefetched(dataset, batch_size=20):
    return dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)


def dummy():
    return tf.data.Dataset.from_tensors(({'image': tf.zeros((1, 224, 224, 3), dtype=tf.float32),
                                          'description': tf.constant([""], dtype=tf.string)},
                                         tf.constant(0, dtype=tf.int64)))


def find_cardinality(dataset):
    i = 0
    for _ in dataset:
        i += 1
    return i


class DatasetHandler(object):
    """
            Holds the tensorflow dataset. Can mask out words.
    """

    def __init__(self, *args, get_dataset=True, **kwargs):
        """
            Arguments: The same arguments as to get_dataset
        """
        self.imagenet_labels_list = inet.get_dataset_labels()
        self.training_descriptions = None
        self.validation_descriptions = None
        self.test_descriptions = None
        self.no_description_available = []

        self._training_set = None
        self._validation_set = None
        self._test_set = None
        if get_dataset:
            self.get_dataset(*args, **kwargs)

        # Used in mask_proportions. Put here so static variables are not necessary
        self.t = tf.Variable(0.0, dtype=tf.float32)
        self.T = tf.Variable(0.0, dtype=tf.float32)
        self.p_min = tf.Variable(0.4, dtype=tf.float32)
        self.p_max = tf.Variable(1.0, dtype=tf.float32)

    def get_dataset(self, train_kb='wordnet', val_kb='wordnet', test_kb='wordnet',
                    train_val_split=0.8,
                    use_reduced_dataset=False,
                    seed=None):
        """
            Creates the training and validation set.  (How about test set?)

            Args:
                train_kb / val_kb - A string signifying which knowledge base to get the hints from in training
                                    and validation. Can be: wikipedia, wordnet or both
                train_test_split - A float signifying the split-percentage
                use_reduced_dataset - A boolean signifying whether to use the reduced imagenet2012 dataset.
        """
        # Import ImageNet2012
        split_pct = int(train_val_split * 100)
        split = [
            "train[0%:{:d}%]".format(split_pct),
            "train[{:d}%:100%]".format(split_pct)
        ]
        training_set, validation_set = tfds.load(
            'imagenet2012_subset' if use_reduced_dataset else 'imagenet2012',
            data_dir='/work3/s184399/imagenet',
            split=split
        )
        test_set = tfds.load(
            'imagenet2012_subset' if use_reduced_dataset else 'imagenet2012',
            data_dir='/work3/s184399/imagenet',
            split='validation'
        )

        self.training_descriptions = self.get_descriptions(knowledge_bases=train_kb)
        self.validation_descriptions = self.get_descriptions(knowledge_bases=val_kb)
        self.test_descriptions = self.get_descriptions(knowledge_bases=test_kb)

        self._training_set = preprocess_dataset(training_set, self.training_descriptions, seed=seed)
        self._validation_set = preprocess_dataset(validation_set, self.validation_descriptions, seed=seed)
        self._test_set = preprocess_dataset(test_set, self.test_descriptions, seed=seed)

        return self._training_set, self._validation_set, self._test_set

    @property
    def training_set(self):
        return self._training_set

    @property
    def validation_set(self):
        return self._validation_set

    @property
    def test_set(self):
        return self._test_set

    def mask_words_in_string(self, s, p=None, decreasing_p=False, seed=None):
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
        p = tf.cond(decreasing_p,
                    true_fn=lambda: (self.p_max - self.p_min) * (self.T - self.t) / self.T,
                    false_fn=lambda: p,
                    name='Decrement_p'
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

        self.t.assign_add(tf.constant(1.0))

        return out

    def mask_words(self, dataset, p=None, decreasing_p=False, epochs=1, seed=None):
        """
            Returns a new dataset in which (1-p)*100% of the words in the descriptions have been removed,
            i.e. p*100% are kept.
            Order is preserved.

            Arguments:
                p - A fixed value for p. A tf.constant. If none is given, it defaults to -1.0, which indicates that
                    p is dynamically chosen.
                dataset - The dataset to be mapped
                decreasing_p - Whether p should instead be decreased throughout the training (for curriculum learning)
                epochs - Used when decreasing_p is True. This just stores how many epochs it should be decreasing over

        """
        if p is None:
            p = tf.constant(-1.0)
        if decreasing_p:
            self.t.assign(0.0)
            self.T.assign(tf.constant(float(epochs)) * tf.cast(dataset.cardinality(), dtype=tf.float32))
        with tf.device('/cpu:0'):
            return dataset.map(
                lambda X, y: ({'image': X['image'],
                               'description': self.mask_words_in_string(p=p,
                                                                        s=X['description'],
                                                                        decreasing_p=tf.constant(decreasing_p),
                                                                        seed=seed),
                               }, y)
            )

    def get_wordnet_descriptions(self):
        """
            Goes through the wordnet taxonomy and obtains the definitions for the classes in imagenet.
        """
        descriptions = []
        for i, label in enumerate(self.imagenet_labels_list):
            desc = wnet.find_synset(label).definition()
            if desc == "" and i not in self.no_description_available:
                self.no_description_available.append(i)
            descriptions.append([desc])
        return descriptions

    def get_wikipedia_descriptions(self):
        """
            Obtains the wikipedia descriptions and returns a list of sentence. If no description for a category is
            available, an empty list is given instead.
        """
        wiki_descriptions = 'wiki_descriptions.json'
        with open(wiki_descriptions, 'r') as f_obj:
            desc_dict = json.load(f_obj)
        descriptions = []
        for i, label in enumerate(self.imagenet_labels_list):
            if label in desc_dict.keys():
                descriptions.append(desc_dict[label].replace('\n', ' ').split('. '))
            else:
                descriptions.append([])
                if i not in self.no_description_available:
                    self.no_description_available.append(i)
        return descriptions

    def get_descriptions(self, knowledge_bases):
        """
            Returns the descriptions from the selected knowledge base(s).
        """
        knowledge_bases = knowledge_bases.lower()
        if knowledge_bases == 'both':
            descriptions = self.get_wordnet_descriptions()
            wiki_descriptions = self.get_wikipedia_descriptions()
            for i in range(len(descriptions)):
                descriptions[i] += wiki_descriptions[i]  # Concatenate lists
            return tf.ragged.constant(descriptions)
        elif knowledge_bases == 'wikipedia':
            return tf.ragged.constant(self.get_wikipedia_descriptions())
        elif knowledge_bases == 'wordnet':
            return tf.convert_to_tensor(self.get_wordnet_descriptions())
        return None

    def drop_observations_without_hints(self, for_all_partitions=False):
        if not for_all_partitions:
            def predicate(X, y):
                return X['description'] != b''
            self._training_set = self._training_set.filter(predicate)
            self._validation_set = self._validation_set.filter(predicate)
            self._test_set = self._test_set.filter(predicate)
        else:
            no_description_available = tf.constant(self.no_description_available, dtype=tf.int64)
            def predicate(X, y):
                return tf.math.reduce_all(y != no_description_available)
            self._training_set = self._training_set.filter(predicate)
            self._validation_set = self._validation_set.filter(predicate)
            self._test_set = self._test_set.filter(predicate)
        return self


def unit_test():
    import numpy as np
    import sys
    """
        Tests some of the functions:
        - mask_words_in_string
        - mask_words
    """
    print("\n##########################" +
          "\n# Performing unit tests. #" +
          "\n##########################\n")

    dataset = DatasetHandler(get_dataset=False)

    # TEST: mask_words_in_string
    a = 'test1 test2 test3 test4 test5'
    a_b = [b'test1', b'test2', b'test3', b'test4', b'test5']
    a_tf = tf.convert_to_tensor(a)
    a_masked = tf.strings.split(dataset.mask_words_in_string(p=tf.constant(0.5), s=a_tf))
    assert tf.size(a_masked).numpy() == 3
    assert tf.size(tf.unique(a_masked)[0]).numpy() == 3
    assert a_masked[0] in a_b
    assert a_masked[1] in a_b
    assert a_masked[2] in a_b
    print("\nTEST PASSED: mask_words_in_string\n")

    # TEST: mask_words
    ds = tf.data.Dataset.range(5).map(lambda x: ({'image': x, 'description': a}, 0))
    ds = dataset.mask_words(p=tf.constant(0.75), dataset=ds)
    i = 0
    for x, y in ds:
        a_masked = tf.strings.split(x['description'])
        assert float(i) == x['image'].numpy()
        assert tf.size(a_masked).numpy() == 4
        assert tf.size(tf.unique(a_masked)[0]).numpy() == 4
        for word in a_masked:
            assert word in a_b
        i += 1
    print("\nTEST PASSED: mask_words\n")

    # TEST: mask_words on the validation set
    _, val, _ = dataset.get_dataset(train_kb='wordnet', val_kb='wordnet',
                                 train_val_split=0.01,
                                 use_reduced_dataset=True)
    # print(val)
    # -> <PrefetchDataset shapes: ({image: (None, 224, 224, 3), description: (None,)}, (None,)),
    #                              types: ({image: tf.float32, description: tf.string}, tf.int64)>
    val = dataset.mask_words(p=tf.constant(0.1), dataset=val)
    for x, _ in val:  # Iterate through each sample in the dataset and see if the descriptions are non-empty
        assert x['description'].numpy() != ""
    print("\nTEST PASSED: mask_words on the validation set (not batched)\n")
    # OK: Masking on the validation set will not produce empty strings.

    # TEST: mask_words on the validation set - produce empty words
    _, val, _ = dataset.get_dataset(train_kb='wordnet', val_kb='both', train_val_split=0.01, use_reduced_dataset=True)
    # print(val)
    # -> <PrefetchDataset shapes: ({image: (None, 224, 224, 3), description: (None,)}, (None,)),
    #                              types: ({image: tf.float32, description: tf.string}, tf.int64)>
    val = dataset.mask_words(p=tf.constant(0.0), dataset=val)
    for x, _ in val:  # Iterate through each sample in the dataset and see if the descriptions are empty
        assert len(x['description'].numpy()) == 0
    print("\nTEST PASSED: mask_words on the validation set (not batched) to produce empty strings\n")

    # TEST: mask_words with dynamic value for p
    _, val, _ = dataset.get_dataset(train_kb='wordnet', val_kb='wordnet', train_val_split=0.01, use_reduced_dataset=True)
    val_masked = dataset.mask_words(dataset=val)
    p_avg = 0
    i = 0
    for x, y in zip(val, val_masked):
        len_unmasked = len(x[0]['description'].numpy())
        len_masked = len(y[0]['description'].numpy())
        p_avg += len_masked / len_unmasked
        i += 1
    p_avg /= i
    print(p_avg)
    assert np.isclose(p_avg, 0.5, atol=0.05)
    print("\nTEST PASSED: mask_words on the validation set with dynamic masking proportion\n")

    # TEST: mask_words with decreasing value for p
    _, val, _ = dataset.get_dataset(train_kb='wordnet', val_kb='wordnet',
                                 train_val_split=0.01, use_reduced_dataset=True)
    val_masked = dataset.mask_words(dataset=val, decreasing_p=True)
    dataset.p_max.assign(1.0)
    dataset.p_min.assign(0.0)
    p_max = np.array(1.0, dtype=np.float32)
    p_min = np.array(0.0, dtype=np.float32)
    t = np.array(0.0, dtype=np.float32)
    T = val.cardinality().numpy().astype(np.float32)
    avg_length = 0.0
    for x, y in zip(val, val_masked):
        len_unmasked = np.array(len(x[0]['description'].numpy().split()), dtype=np.float32)
        len_masked = np.array(len(y[0]['description'].numpy().split()), dtype=np.float32)
        avg_length += len_unmasked
        target_length = np.ceil(((p_max - p_min) * (T - t) / T) * len_unmasked)
        assert target_length == len_masked
        t += 1.0
    print("\nTEST PASSED: mask_words on the validation set with decreasing masking proportion")
    print("AVERAGE DEFINITION LENGTH: {}\n".format(avg_length / T))

    # Test if the seeded mask_words give the same result when queried twice.
    # Note that the mask_words-function produces a dataset in which the entries
    # are computed ad hoc using on a random generator. Here we test that this
    # random generator is indeed seeded correctly.
    _, val_set, _ = dataset.get_dataset(train_kb='wordnet',
                                     val_kb='both',
                                     train_val_split=0.01,
                                     use_reduced_dataset=True,
                                     seed=42)
    val_set_masked = dataset.mask_words(p=tf.constant(0.6), dataset=val_set, seed=42)
    a = list(val_set_masked.map(lambda x, y: x['description']).as_numpy_iterator())
    b = list(val_set_masked.map(lambda x, y: x['description']).as_numpy_iterator())
    assert a == b
    print("\nTEST PASSED: seeded mask_words gives the same result twice")

    # Test that the wikipedia descriptions and wordnet descriptions are not the same,
    # and that 'both' is not the same as either of them for all elements
    _, valboth, _ = dataset.get_dataset(train_kb='wordnet', val_kb='both', train_val_split=0.01,
                                     use_reduced_dataset=True)
    _, valwiki, _ = dataset.get_dataset(train_kb='wordnet', val_kb='wikipedia', train_val_split=0.01,
                                     use_reduced_dataset=True)
    _, valwnet, _ = dataset.get_dataset(train_kb='wordnet', val_kb='wordnet', train_val_split=0.01,
                                     use_reduced_dataset=True)
    all_the_same = True
    for both, wiki, wnet in zip(valboth, valwiki, valwnet):
        assert wiki[0]['description'] != wnet[0]['description']
        all_the_same &= (both[0]['description'] == wiki[0]['description']) or \
                        (both[0]['description'] == wnet[0]['description'])
    assert not all_the_same
    print("\nTEST PASSED: using both wikipedia and wordnet\n")

    # Tests if we can drop all the observations with no description
    dataset.get_dataset(train_kb='wikipedia', val_kb='wikipedia', train_val_split=0.01,
                        use_reduced_dataset=True)
    original_cardinality = dataset.validation_set.cardinality().numpy()
    dataset.drop_observations_without_hints()
    valwiki = dataset.validation_set
    new_cardinality = 0
    for X, y in valwiki:
        assert X['description'].numpy() != b''
        new_cardinality += 1
    print("\nTEST PASSED: dropped all observations without hints successfully")
    print("Dataset length reduced from {} to {}.\n".format(original_cardinality,
                                                           new_cardinality))

    # Tests if we can drop all observations across the partitions, if the class has no description in one of the used
    # knowledge bases
    dataset.get_dataset(train_kb='wikipedia', val_kb='wordnet', test_kb='wordnet', train_val_split=0.8,
                        use_reduced_dataset=True)
    dataset.drop_observations_without_hints(for_all_partitions=True)
    train = dataset.training_set
    val = dataset.validation_set
    test = dataset.test_set
    no_descriptions = dataset.no_description_available
    for X, y in train:
        assert X['description'] != b''
    for part in [val, test]:
        for X, y in part:
            assert int(y.numpy()) not in no_descriptions
            assert X['description'] != b''
    print("TEST PASSED: dropped categories which did not have hints in all partitions")


if __name__ == '__main__':
    unit_test()

