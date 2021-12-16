import tensorflow as tf
import tensorflow_hub as hub
import tensorflow_text as text
import os
import numpy as np

from networks import network_architectures


def get_bert(final_op='pooled_output'):
    # Source: https://www.tensorflow.org/text/tutorials/classify_text_with_bert
    tfhub_handle_encoder = 'https://tfhub.dev/tensorflow/bert_en_cased_L-12_H-768_A-12/3'
    tfhub_handle_preprocess = 'https://tfhub.dev/tensorflow/bert_en_cased_preprocess/3'
    text_input = tf.keras.layers.Input(shape=(), dtype=tf.string, name="text_input")
    preprocessing = hub.KerasLayer(tfhub_handle_preprocess, name="preprocessing", trainable=False)
    encoder = hub.KerasLayer(tfhub_handle_encoder, name="BERT_encoder", trainable=False)
    bert_raw = encoder(preprocessing(text_input))
    if final_op == 'pooled_output':
        # Next sentence predictor (embedding of the CLS token)
        bert = tf.keras.Model(text_input,
                              bert_raw['pooled_output'])
    elif final_op in ['avg_sequence', 'average_sequence', 'average', 'avg sequence', 'average sequence']:
        # Average of sequence outputs
        bert = tf.keras.Model(text_input,
                              tf.reduce_mean(bert_raw['sequence_output'], axis=1))
    else:
        bert = tf.keras.Model(text_input,
                              bert_raw)
    return bert


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
                 bert_final_op='pooled_output',
                 **kwargs):
        super(ConcatCombinedClassifier, self).__init__(**kwargs)

        if im_net is None:
            im_net = network_architectures['resnet50']['net']

        # Construct the image classification part
        image_clf = im_net(classifier_activation=None)  # Get the network
        self.image_clf = tf.keras.Model(
            inputs=image_clf.inputs,
            outputs=image_clf.layers[-2].output
        )  # Discard the final dense layer.
        self.image_clf.trainable = False

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
                 bert_final_op='pooled_output',
                 **kwargs):
        super(AddCombinedClassifier, self).__init__(**kwargs)
        self.log_vectors = log_vectors
        self.logged_vectors = []

        if im_net is None:
            im_net = network_architectures['resnet50']['net']

        # Construct the image classification part
        self.image_clf = im_net(classifier_activation=None)  # Get the network
        self.image_clf.trainable = False

        # BERT part of the network
        self.bert = get_bert(final_op=bert_final_op)
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


class WeightedSumCombinedClassifier(tf.keras.Model):
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
                 bert_final_op='pooled_output',
                 **kwargs):
        super(WeightedSumCombinedClassifier, self).__init__(**kwargs)
        self.log_vectors = log_vectors
        self.logged_vectors = []

        if im_net is None:
            im_net = network_architectures['resnet50']['net']

        # Construct the image classification part
        self.image_clf = im_net(classifier_activation=None)  # Get the network
        self.image_clf.trainable = False

        # BERT part of the network
        self.bert = get_bert(final_op=bert_final_op)
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

        # Dial
        self.dial = tf.Variable(0.0, trainable=True, dtype=tf.float32, name='dial')
        self.one = tf.constant(1, dtype=tf.float32)

    def call(self, x, training=False):
        img, txt = x['image'], x['description']  # Maybe not the way to go
        img_output = self.image_clf(img)
        img_output_normalized = tf.linalg.normalize(img_output, axis=1)[0]
        txt_embedding = self.bert(txt)
        txt_output = self.dense_layers(txt_embedding, training=training)
        txt_output_normalized = tf.linalg.normalize(txt_output, axis=1)[0]
        dial = tf.nn.sigmoid(self.dial) # self.dial #tf.nn.sigmoid(self.dial)

        return tf.nn.softmax(dial * img_output_normalized
                             + (self.one - dial)*txt_output_normalized)


class ResNet50Baseline(tf.keras.Model):
    """
        The usual ResNet50 model for baseline. This model should not be trained.
    """

    def __init__(self,
                 im_net=None,
                 dense_layer_units=None,
                 n_classes=1000,
                 kernel_regularizer=None,
                 log_vectors=False,
                 bert_final_op='pooled_output',
                 **kwargs):
        super(ResNet50Baseline, self).__init__(**kwargs)

        im_net = network_architectures['resnet50']['net']

        # Construct the image classification part
        self.image_clf = im_net(classifier_activation="softmax")  # Get the network
        self.image_clf.trainable = False

    def call(self, x, training=False):
        return self.image_clf(x['image'])


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
        txt = x['description']
        txt_embedding = self.bert(txt)
        txt_output = self.dense_layers(txt_embedding, training=training)

        return tf.nn.softmax(txt_output)
