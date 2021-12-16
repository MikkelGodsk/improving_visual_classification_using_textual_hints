"""
    Experiment:
        We would like to try and see if feeding parts of the wordnet definitions into a separate network can aid a
        pre-trained ResNet50 model in improving the ImageNet classification.

        This will be done by obtaining the WordNet definitions, masking words and creating embeddings using spaCy.
        This embedding will be fed into a feedforward network, with which the output is concatenated to the ResNet50
        output. Finally, softmax is applied in order to obtain the top-N predictions.

    Data:
        We load the imagenet2012 dataset (or subset) and create a new validation set thereof. The original validation
        set we will be using as our test set later on.

        NOTE: The TFDS documentation guarantees that the datasets are split deterministically.
        NOTE: The split between the training set and validation set are NOT stratified. But since they contain a lot of
            observations for each class, this might be a minor issue.
"""
import wordnet_interface as wnet
import imagenet_interface as inet
from networks import network_architectures
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import tensorflow_datasets as tfds

from typing import Callable
import os


def get_bert():
    # Source: https://www.tensorflow.org/text/tutorials/classify_text_with_bert
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3'
    tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3'
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text_input")
    preprocessing = hub.KerasLayer(tfhub_handle_preprocess, name="preprocessing", trainable=False)
    encoder = hub.KerasLayer(tfhub_handle_encoder, name="BERT_encoder", trainable=False)
    return tf.keras.Model(text_input,
                          encoder(preprocessing(text_input))['pooled_output'])  # What kind of pooling??


def get_resnet50_no_top():
    im_net = network_architectures['resnet50']['net']

    # Construct the image classification part
    image_clf = im_net()  # Get the network
    image_clf = tf.keras.Model(
        inputs=image_clf.inputs,
        outputs=image_clf.layers[-2].output
    )  # Discard the final dense layer.
    image_clf.trainable = False

    return image_clf


def get_resnet50_with_top():
    im_net = network_architectures['resnet50']['net']

    # Construct the image classification part
    image_clf = im_net()  # Get the network
    image_clf.layers[-1].activation = None
    image_clf = tf.keras.Model(
        inputs=image_clf.inputs,
        outputs=image_clf.layers[-1].output
    )  # Discard the final dense layer.
    image_clf.trainable = False

    return image_clf


class ConcatCombinedClassifier(tf.keras.Model):
    def __init__(self, im_net=None,
                 dense_layer_units=None,
                 n_classes=1000,
                 kernel_regularizer=None,
                 **kwargs):
        super(ConcatCombinedClassifier, self).__init__(**kwargs)

        if im_net is None:
            im_net = network_architectures['resnet50']['net']

        # Construct the image classification part
        image_clf = im_net()  # Get the network
        self.image_clf = tf.keras.Model(
            inputs=image_clf.inputs,
            outputs=image_clf.layers[-2].output
        )  # Discard the final dense layer.
        self.image_clf.trainable = False

        # BERT part of the network
        self.bert = get_bert()

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

        # Final dense layer
        self.concat = tf.keras.layers.Concatenate()
        self.final_dense = tf.keras.layers.Dense(units=n_classes,
                                                 trainable=True,
                                                 kernel_regularizer=kernel_regularizer)

    def call(self, x, training=False):
        img, txt = x['image'], x['description']  # Maybe not the way to go
        img_output = self.image_clf(img)
        txt_embedding = self.bert(txt)
        txt_output = self.dense_layers(txt_embedding, training=training)

        return tf.nn.softmax(self.final_dense(self.concat([img_output, txt_output]), training=training))


class AddCombinedClassifier(tf.keras.Model):
    """
        A classifier using an image and a piece of text for classification. Here, we add the output of BERT and ResNet50
        for the final classification.

        TODO: Figure out how to sum up the contributions from BERT and ResNet50. Softmax is scale-sensitive, so it will
        matter whether we multiply by ½ or not. And also keep in mind that we do not at all retrain BERT nor ResNet50.
        Only the single dense layer following after BERT.
    """

    def __init__(self,
                 im_net=None,
                 dense_layer_units=None,
                 n_classes=1000,
                 kernel_regularizer=None,
                 log_vectors=False,
                 **kwargs):
        super(AddCombinedClassifier, self).__init__(**kwargs)
        self.log_vectors = log_vectors
        self.logged_vectors = []

        if im_net is None:
            im_net = network_architectures['resnet50']['net']

        # Construct the image classification part
        self.image_clf = im_net(classifier_activation=False)  # Get the network
        self.image_clf.trainable = False

        # BERT part of the network
        self.bert = get_bert()
        self.bert.trainable = False

        # Dense layers before concatenation
        self.dense_layers = tf.keras.Sequential([])  # tf.keras.layers.Dropout(rate=0.3)])  # Can it be annealed?
        if dense_layer_units is not None:
            for units in dense_layer_units:
                self.dense_layers.add(tf.keras.layers.Dense(units=units,
                                                            trainable=True,
                                                            kernel_regularizer=kernel_regularizer))
                self.dense_layers.add(tf.keras.layers.BatchNormalization())
                self.dense_layers.add(tf.keras.layers.ReLU())
                # self.dense_layers.add(tf.keras.layers.Dropout(rate=0.2))

        # Output layer
        self.dense_layers.add(tf.keras.layers.Dense(units=n_classes,
                                                    trainable=True,
                                                    kernel_regularizer=kernel_regularizer))

        # Final layer
        self.add = tf.keras.layers.Add()

    def call(self, x, training=False):
        img, txt = x['image'], x['description']  # Maybe not the way to go
        img_output = self.image_clf(img)
        txt_embedding = self.bert(txt)
        txt_output = self.dense_layers(txt_embedding, training=training)
        if self.log_vectors:
            self.logged_vectors.append(txt_output)

        return tf.nn.softmax((self.add([img_output, txt_output])))


def pred_contains_label(pred_list, true_label):
    """
        To see if the predicted list (e.g. top 5) contains the true label.
    """
    for p in pred_list:
        if p[0] == true_label:
            return True
    return False


class HintsExperiment(object):
    """
        Just keeping all the functions in a class, so the variables can be exchanged more nicely.
    """

    def __init__(self, batch_size=20, directory='/work3/s184399/hints_experiment'):  # use_reduced_dataset=False,
        with tf.device('/cpu:0'):
            self.combined_model = None
            self.labels = inet.get_dataset_labels()
            self.preprocess_img_input_fn: Callable
            self.batch_size = batch_size

            self.im_net_pkg = network_architectures['resnet50']
            self.preprocess_img_input_fn = self.im_net_pkg['preprocess_input']
            self.decode_predictions = self.im_net_pkg['decode_predictions']

            # Get the list with synset names corresponding to the categorical encoding of the labels
            self.imagenet_labels_list = inet.get_dataset_labels()  # Synset names for the binary encoding.
            self.wordnet_descriptions = self._create_wordnet_descriptions()

            # Get training set and create a new validation set
            # self.get_dataset(
            #    use_reduced_dataset=use_reduced_dataset, preprocess=True
            # )

            self.train_loss = []
            self.train_acc = []
            self.val_loss = []
            self.val_acc = []

            self.experiment_results = []

            self.directory = directory
            if not os.path.isdir(self.directory):
                os.makedirs(self.directory)

            # Used in mask_proportions. Put here so static variables are not necessary
            self.t = tf.Variable(0.0, dtype=tf.float32)
            self.T = tf.Variable(0.0, dtype=tf.float32)
            self.p_min = tf.Variable(0.4, dtype=tf.float32)
            self.p_max = tf.Variable(1.0, dtype=tf.float32)

    def get_dataset(self, train_test_split=0.8, use_reduced_dataset=False, preprocess=True):
        """
            Fetches the training set and validation set. If preprocess is True, then it will do the preprocessing as
            well, i.e. adding text, preprocessing the images for ResNet50, making batches etc.
        """
        split_pct = int(train_test_split * 100)
        split = [
            "train[0%:{:d}%]".format(split_pct),
            "train[{:d}%:100%]".format(split_pct)
        ]
        self.training_set, self.validation_set = tfds.load(
            'imagenet2012_subset' if use_reduced_dataset else 'imagenet2012',
            data_dir='/work3/s184399/imagenet',
            split=split
        )
        if preprocess:
            self.training_set = self._preprocess_dataset(self.training_set)
            self.validation_set = self._preprocess_dataset(self.validation_set)

        print("Training set length: {}\nValidation set length: {}".format(
            len(self.training_set),
            len(self.validation_set)
        )
        )

        return self.training_set, self.validation_set

    def _create_wordnet_descriptions(self):
        """
            Goes through the wordnet taxonomy and obtains the definitions for the classes in imagenet.
        """
        descriptions = []
        for label in self.imagenet_labels_list:
            descriptions.append(wnet.find_synset(label).definition())
        return tf.convert_to_tensor(descriptions)

    def _prepare_sample(self, sample):
        """
            Prepares a sample (an image) from the dataset by adding a description and the label.
        """
        with tf.device('/cpu:0'):
            img = self.preprocess_img_input_fn(tf.image.resize(sample['image'], (224, 224)))
            # img = tf.expand_dims(img, axis=0)

            return {'image': img, 'description': self.wordnet_descriptions[sample['label']]}, sample['label']

    def _preprocess_dataset(self, dataset):
        """
            Prepares all observations in the dataset and enables batch and prefetch.
        """
        return dataset.map(self._prepare_sample)

    def _make_batched_and_prefetched(self, dataset):
        return dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)

    def new_model(self, regularization_rate=0.0, combination_op='addition'):
        """
            Creates and compiles the model.
        """
        # self.net = ConcatCombinedClassifier(im_net=self.im_net_pkg['net'], dense_layer_units=None)  # [300] * 2)
        if combination_op == 'addition':
            self.combined_model = AddCombinedClassifier(
                kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))
        else:
            self.combined_model = ConcatCombinedClassifier(
                kernel_regularizer=tf.keras.regularizers.l2(regularization_rate))
        self.combined_model.compile(optimizer='adam',
                                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                    metrics=[tf.keras.metrics.SparseTopKCategoricalAccuracy(k=1)])
        self.build_model()
        return self.combined_model

    def has_model(self):
        return self.combined_model is not None

    def train_model(self, epochs=2, dynamic_masking=False, dynamic_p=None, decreasing_p=False, validation=True):
        """
            Trains the model and stores the learning curves.

            Call .get_dataset with the specific parameters first!
        """
        # Training procedure
        if not self.has_model():
            self.new_model()

        if dynamic_masking:
            train_data = self._mask_words(dataset=self.training_set, p=dynamic_p, decreasing_p=decreasing_p)
            val_data = self._mask_words(dataset=self.validation_set, p=dynamic_p, decreasing_p=decreasing_p)
        else:
            train_data = self.training_set
            val_data = self.validation_set

        train_data = self._make_batched_and_prefetched(train_data)
        val_data = self._make_batched_and_prefetched(val_data)
        hist = self.combined_model.fit(x=train_data,
                                       epochs=epochs,
                                       validation_data=val_data if validation else None,
                                       verbose=2,
                                       )
        self.train_loss += hist.history['loss']  # Concatenate to loss history
        self.train_acc += hist.history['sparse_top_k_categorical_accuracy']
        if validation:
            self.val_loss += hist.history['val_loss']
            self.val_acc += hist.history['val_sparse_top_k_categorical_accuracy']

    def evaluate(self, dynamic_masking=False, dynamic_p=None):
        """
            Evaluates the model on the batched validation set.

            Call .get_dataset with the specific parameters first!
        """
        if dynamic_masking:
            val_data = self._mask_words(dataset=self.validation_set, p=dynamic_p)
        else:
            val_data = self.validation_set

        val_data = self._make_batched_and_prefetched(val_data)
        return self.combined_model.evaluate(x=val_data, verbose=2)

    def save_model(self):  # , model_weights_path='/work3/s184399/DIRECTORY/model_weights'):
        """
            Saves the model. If the model has not been created, it throws an error.
        """
        assert self.has_model()
        # model_weights_path = model_weights_path.replace('DIRECTORY', self.directory)
        # self.net.save_weights(model_weights_path)
        self.combined_model.save_weights(os.path.join(self.directory, 'model_weights'))

    def save_model_and_learning_curves(self, model_weights_path='/work3/s184399/DIRECTORY/model_weights'):
        """
            Saves the model and learning curves
        """
        # Status: Tested
        # model_weights_path = model_weights_path.replace('DIRECTORY', self.directory)
        # TODO: Save classifications
        np.save(os.path.join(self.directory, 'model_train_cross_entropy_loss'), self.train_loss)
        np.save(os.path.join(self.directory, 'model_train_top_K_accuracy'), self.train_acc)
        np.save(os.path.join(self.directory, 'model_validation_cross_entropy_loss'), self.val_loss)
        np.save(os.path.join(self.directory, 'model_validation_top_K_accuracy'), self.val_acc)
        self.save_model()

    def build_model(self):
        """
            We just run a simple observation through the model to ensure it has been built
        """
        ds = tf.data.Dataset.from_tensors(({'image': tf.zeros((1, 224, 224, 3), dtype=tf.float32),
                                            'description': tf.constant([""], dtype=tf.string)},
                                           tf.constant(0, dtype=tf.int64)))
        self.combined_model.predict(ds)

    def load_model(self, model_weights_path=None):
        """
            Loads a model from the weights. If no path is given, it uses the one given in the constructor

            Before calling this function, be sure to invoke "new_model".
        """
        # Status: Not tested, but code seems to work.
        assert self.has_model()  # Model should be defined first
        if model_weights_path is None:
            model_weights_path = os.path.join(self.directory, 'model_weights')

        return self.combined_model.load_weights(model_weights_path)
        # status.assert_consumed()  # Did the model load properly?

    def _mask_words_aux(self, s, p=None, decreasing_p=False, seed=None):
        """
            An auxiliary function for _mask_words.
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

    def _mask_words(self, p=None, dataset=None, decreasing_p=False, epochs=1, seed=None):
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
                               'description': self._mask_words_aux(p=p,
                                                                   s=X['description'],
                                                                   decreasing_p=tf.constant(decreasing_p),
                                                                   seed=seed),
                               }, y)
            )

    def mask_proportion_experiment(self):
        """
            Runs the experiment where we try different (fixed) proportions for dynamic masking.
            Returns the accuracy on the validation set with masked hints.
            p is the proportion of kept words.
        """
        inc = 0.1
        for p in np.arange(0.0, 1 + inc, inc, dtype=np.float32):
            masked_ds = self._mask_words(p=tf.constant(p),
                                         dataset=self.validation_set)
            masked_ds = self._make_batched_and_prefetched(masked_ds)
            self.experiment_results.append(self.combined_model.evaluate(masked_ds, verbose=2)[1])

        return self.experiment_results

    def regularization_rate_experiment(self, rates, combination_op='addition'):
        """
            Tries out different regularization rates. Trains for 4 epochs, then validates on the (large) validation set.

            Inputs:
            - rates : A list of floating point numbers specifying the regularization rates.

            Outputs:
            - The accuracies on the validation set.
        """
        self.get_dataset(train_test_split=0.1,
                         use_reduced_dataset=False)  # Make a much smaller dataset for faster training
        for rate in rates:
            self.new_model(regularization_rate=rate, combination_op=combination_op)
            self.train_model(epochs=4, dynamic_masking=True, validation=False)  # Don't do validation here.
            # It'll take forever
            acc = self.evaluate(dynamic_masking=True)[1]  # Slooow: Large validation set.
            self.experiment_results.append(acc)
            print("Regularization rate: {} - Accuracy: {}".format(rate, acc))

        return self.experiment_results

    def save_experiment_results(self):
        """
            If the experiment has been run, it saves the results (accuracies) to a file.
            If not, it throws an error.
        """
        assert len(self.experiment_results) > 0
        path = os.path.join(self.directory, 'experiment_results')
        np.save(path, self.experiment_results)

    def unit_test(self):
        """
            Tests some of the functions:
            - _mask_words_aux
            - _mask_words
        """
        print("\n##########################" +
              "\n# Performing unit tests. #" +
              "\n##########################\n")

        # TEST: _mask_words_aux
        a = 'test1 test2 test3 test4 test5'
        a_b = [b'test1', b'test2', b'test3', b'test4', b'test5']
        a_tf = tf.convert_to_tensor(a)
        a_masked = tf.strings.split(self._mask_words_aux(p=tf.constant(0.5), s=a_tf))
        assert tf.size(a_masked).numpy() == 3
        assert tf.size(tf.unique(a_masked)[0]).numpy() == 3
        assert a_masked[0] in a_b
        assert a_masked[1] in a_b
        assert a_masked[2] in a_b
        print("\nTEST PASSED: _mask_words_aux\n")

        # TEST: _mask_words
        ds = tf.data.Dataset.range(5).map(lambda x: ({'image': x, 'description': a}, 0))
        ds = self._mask_words(p=tf.constant(0.75), dataset=ds)
        i = 0
        for x, y in ds:
            a_masked = tf.strings.split(x['description'])
            assert float(i) == x['image'].numpy()
            assert tf.size(a_masked).numpy() == 4
            assert tf.size(tf.unique(a_masked)[0]).numpy() == 4
            for word in a_masked:
                assert word in a_b
            i += 1
        print("\nTEST PASSED: _mask_words\n")

        # TEST: _mask_words on the validation set
        _, val = self.get_dataset(train_test_split=0.8, use_reduced_dataset=False, preprocess=True)
        # print(val)  # -> <PrefetchDataset shapes: ({image: (None, 224, 224, 3), description: (None,)}, (None,)), types: ({image: tf.float32, description: tf.string}, tf.int64)>
        val = self._mask_words(p=tf.constant(0.1), dataset=val)
        for x, _ in val:  # Iterate through each sample in the dataset and see if the descriptions are non-empty
            assert x['description'].numpy() != ""
        print("\nTEST PASSED: _mask_words on the validation set (not batched)\n")
        # OK: Masking on the validation set will not produce empty strings.

        # TEST: _mask_words on the validation set - produce empty words
        _, val = self.get_dataset(train_test_split=0.8, use_reduced_dataset=False, preprocess=True)
        # print(val)  # -> <PrefetchDataset shapes: ({image: (None, 224, 224, 3), description: (None,)}, (None,)), types: ({image: tf.float32, description: tf.string}, tf.int64)>
        val = self._mask_words(p=tf.constant(0.0), dataset=val)
        for x, _ in val:  # Iterate through each sample in the dataset and see if the descriptions are empty
            assert len(x['description'].numpy()) == 0
        print("\nTEST PASSED: _mask_words on the validation set (not batched) to produce empty strings\n")

        # TEST: _mask_words with dynamic value for p
        _, val = self.get_dataset(train_test_split=0.8, use_reduced_dataset=False, preprocess=True)
        val_masked = self._mask_words(dataset=val)
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
        print("\nTEST PASSED: _mask_words on the validation set with dynamic masking proportion\n")

        # TEST: _mask_words with decreasing value for p
        _, val = self.get_dataset(train_test_split=0.8, use_reduced_dataset=True, preprocess=True)
        val_masked = self._mask_words(dataset=val, decreasing_p=True)
        self.p_max.assign(1.0)
        self.p_min.assign(0.0)
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
        print("\nTEST PASSED: _mask_words on the validation set with decreasing masking proportion")
        print("AVERAGE DEFINITION LENGTH: {}\n".format(avg_length / T))

        # Test if the seeded _mask_words give the same result when queried twice. Note that the _mask_words-function
        # produces a dataset in which the entries are computed ad hoc using on a random generator. Here we test that
        # this random generator is indeed seeded correctly.
        _, val_set = self.get_dataset()
        val_set_masked = self._mask_words(p=tf.constant(0.6), dataset=val_set, seed=42)
        a = list(val_set_masked.map(lambda x, y: x['description']).as_numpy_iterator())
        b = list(val_set_masked.map(lambda x, y: x['description']).as_numpy_iterator())
        assert a == b
        print("\nTEST PASSED: seeded _mask_words gives the same result twice")

    def bert_experiment(self):
        """
            Experiment description:
                We run the masked inputs through BERT and store the output vectors. We also store the classification.

                The maskings that will be used are p=0.7 and p=0.2 on the validation set.
        """
        bert = get_bert()
        self.get_dataset(train_test_split=0.2)

        # Create the masked datasets in which we discard the images.
        ds1 = self._make_batched_and_prefetched(
            self._mask_words(p=tf.constant(0.2), dataset=self.training_set).map(lambda X, y: (X['description'],))
        )
        ds2 = self._make_batched_and_prefetched(
            self._mask_words(p=tf.constant(0.7), dataset=self.training_set).map(lambda X, y: (X['description'],))
        )

        self.experiment_results.append(bert.predict(ds1))
        self.experiment_results.append(bert.predict(ds2))
        self.save_experiment_results()

    def correct_bert_experiment(self):
        """
            Experiment description:
                We run the masked inputs through BERT and store the output vectors. We also store the classification.
                The outputs are the BERT embeddings run through the following dense layers

                The maskings that will be used are p=0.7 and p=0.2 on the validation set.
        """
        self.get_dataset(train_test_split=0.2)
        self.combined_model.log_vectors = True

        # Create the masked datasets in which we discard the images.
        ds1 = self._make_batched_and_prefetched(
            self._mask_words(p=tf.constant(0.2), dataset=self.training_set)
        )
        ds2 = self._make_batched_and_prefetched(
            self._mask_words(p=tf.constant(0.7), dataset=self.training_set)
        )

        self.experiment_results.append(self.combined_model.predict(ds1).numpy())
        self.experiment_results.append([
            res.numpy() for res in self.combined_model.logged_vectors
        ])  # Doesn't follow the Liskov substitution principle = Bad
        self.combined_model.logged_vectors = []
        self.experiment_results.append(self.combined_model.predict(ds2).numpy())
        self.experiment_results.append([
            res.numpy() for res in self.combined_model.logged_vectors
        ])
        self.save_experiment_results()

    def resnet50_with_top_experiment(self):
        """
            Here we just log the embeddings from the ResNet50. The outputs will be what we feed into the final softmax.
        """
        resnet = get_resnet50_with_top()
        self.get_dataset(train_test_split=0.2)

        ds = self._make_batched_and_prefetched(
            self.training_set.map(lambda X, y: (X['image'],))
        )

        self.experiment_results.append(resnet.predict(ds))
        self.save_experiment_results()

    def resnet50_no_top_experiment(self):
        """
            Here we just log the outputs from the ResNet50. The outputs will be the embeddings.
        """
        resnet = get_resnet50_no_top()
        self.get_dataset(train_test_split=0.2)

        ds = self._make_batched_and_prefetched(
            self.training_set.map(lambda X, y: (X['image'],))
        )

        self.experiment_results.append(resnet.predict(ds))
        self.save_experiment_results()

    def correction_experiment(self):
        """
            Runs the experiment where we find predictions that were incorrect with ResNet50 but correct with the
            combined model.

            In order to conduct this experiment, we store all the top-1 predictions of resnet50 and the combined model
            as well as the hint the combined model was given.
        """
        self.get_dataset()
        val_set = self._mask_words(dataset=self.validation_set, seed=42, p=tf.constant(0.7))
        val_no_text = self._make_batched_and_prefetched(
            val_set.map(lambda X, y: X['image'])
        )
        val_set = self._make_batched_and_prefetched(val_set)
        resnet50 = network_architectures['resnet50']['net']()
        resnet_predictions = resnet50.predict(val_no_text)
        combined_model_predictions = self.combined_model.predict(val_set)
        text = []
        correct_labels = []
        for x in val_set:
            text.append(x[0]['description'])
            correct_labels.append(x[1])

        self.experiment_results = [resnet_predictions,
                                   combined_model_predictions,
                                   np.array(text),
                                   np.array(correct_labels)]

        self.save_experiment_results()


if __name__ == '__main__':
    experiment = HintsExperiment(
        directory='/work3/s184399/trained_models/addition_dynamic_masking_70pct_regularization_1e-6'
    )
    # experiment.unit_test()

    #experiment.get_dataset()
    #experiment.regularization_rate_experiment([1e-7, 1e-6, 1e-5], combination_op='addition')
    # experiment.save_experiment_results()

    ## Train model and save weights
    experiment.get_dataset()
    experiment.new_model(combination_op='addition', regularization_rate=1e-6)
    experiment.train_model(epochs=2, dynamic_masking=True, dynamic_p=tf.constant(0.7), validation=True)
    experiment.save_model_and_learning_curves()

    # experiment.load_model()
    # print(experiment.mask_proportion_experiment())
    # experiment.save_experiment_results()

    # experiment.bert_experiment()
    # experiment.resnet50_with_top_experiment()

    #experiment.new_model(combination_op='addition')
    #experiment.load_model(
    #    model_weights_path='/work3/s184399/trained_models/addition_dynamic_masking_70pct_regularization_1e-4'
    #)   # TODO: Test the load_model method.  What is this???
    #experiment.correct_bert_experiment()#correction_experiment()
"""
    Get data from HPC:
scp s184399@transfer.gbar.dtu.dk:~/bscproj/hints_experiment/* '/mnt/c/Users/mikke/OneDrive/Dokumenter/DTU documents/7. semester/Bachelor projekt/TF HPC GPU prediction/hints_experiment'

    Transfer code to HPC:
scp * s184399@transfer.gbar.dtu.dk:~/bscproj

"""
