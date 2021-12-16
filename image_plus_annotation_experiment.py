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
from dataset import DatasetHandler, make_batched_and_prefetched, dummy
from models import *

import numpy as np
import tensorflow as tf
import os


class HintsExperiment(object):
    """
        Just keeping all the functions in a class, so the variables can be exchanged more nicely.
    """

    def __init__(self, batch_size=20, directory='/work3/s184399/hints_experiment'):  # use_reduced_dataset=False,
        with tf.device('/cpu:0'):
            self.combined_model = None
            self.dataset = None

            self.im_net_pkg = network_architectures['resnet50']

            self.train_loss = []
            self.train_acc = []
            self.val_loss = []
            self.val_acc = []

            self.experiment_results = []

            self.directory = directory
            if not os.path.isdir(self.directory):
                os.makedirs(self.directory)

    def get_dataset(self, *args, **kwargs):
        """
            Fetches the training set and validation set. If preprocess is True, then it will do the preprocessing as
            well, i.e. adding text, preprocessing the images for ResNet50, making batches etc.
        """
        self.dataset = DatasetHandler(*args, **kwargs)
        print("Training set length: {}\nValidation set length: {}".format(
            self.dataset.training_set.cardinality().numpy().astype(np.float32),
            self.dataset.validation_set.cardinality().numpy().astype(np.float32)
        )
        )

        return self.dataset.training_set, self.dataset.validation_set

    def new_model(self, regularization_rate=0.0, combination_op='addition', bert_final_op='pooled_output'):
        """
            Creates and compiles the model.
        """
        # self.net = ConcatCombinedClassifier(im_net=self.im_net_pkg['net'], dense_layer_units=None)  # [300] * 2)
        if combination_op == 'addition':
            self.combined_model = AddCombinedClassifier(
                kernel_regularizer=tf.keras.regularizers.l2(regularization_rate),
                bert_final_op=bert_final_op
            )
        elif combination_op == 'concatenation':
            self.combined_model = ConcatCombinedClassifier(
                kernel_regularizer=tf.keras.regularizers.l2(regularization_rate),
                bert_final_op=bert_final_op
            )
        elif combination_op == 'weighted sum':
            self.combined_model = WeightedSumCombinedClassifier(
                kernel_regularizer=tf.keras.regularizers.l2(regularization_rate),
                bert_final_op=bert_final_op
            )
        elif combination_op == 'baseline':
            self.combined_model = ResNet50Baseline()
        self.combined_model.compile(optimizer='adam',
                                    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                                    metrics=[
                                        tf.keras.metrics.SparseTopKCategoricalAccuracy(
                                            k=5,
                                            name='sparse_top_5_categorical_accuracy'
                                        ),
                                        tf.keras.metrics.SparseTopKCategoricalAccuracy(
                                            k=1,
                                            name='sparse_top_1_categorical_accuracy'
                                        )])
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
            train_data = self.dataset.mask_words(dataset=self.dataset.training_set,
                                                 p=dynamic_p,
                                                 decreasing_p=decreasing_p)
            val_data = self.dataset.mask_words(dataset=self.dataset.validation_set,
                                               p=dynamic_p,
                                               decreasing_p=decreasing_p)
        else:
            train_data = self.dataset.training_set
            val_data = self.dataset.validation_set

        train_data = make_batched_and_prefetched(train_data)
        val_data = make_batched_and_prefetched(val_data)
        hist = self.combined_model.fit(x=train_data,
                                       epochs=epochs,
                                       validation_data=val_data if validation else None,
                                       verbose=2,
                                       )
        self.train_loss += hist.history['loss']  # Concatenate to loss history
        self.train_acc += (hist.history['sparse_top_5_categorical_accuracy'],
                           hist.history['sparse_top_1_categorical_accuracy'])
        if validation:
            self.val_loss += hist.history['val_loss']
            self.val_acc += (hist.history['val_sparse_top_5_categorical_accuracy'],
                             hist.history['val_sparse_top_1_categorical_accuracy'])

    def evaluate(self, dynamic_masking=False, dynamic_p=None, verbose=2, evaluate_on='validation'):
        """
            Evaluates the model on the batched validation set.

            Call .get_dataset with the specific parameters first!
        """
        if evaluate_on == 'validation':
            data = self.dataset.validation_set
        elif evaluate_on == 'test':
            data = self.dataset.test_set
        else:
            raise ValueError('Parameter: evaluate_on should be either \'validation\' or \'test\' in .evaluate')

        if dynamic_masking:
            val_data = self.dataset.mask_words(dataset=data, p=dynamic_p)
        else:
            val_data = data

        val_data = make_batched_and_prefetched(val_data)
        return self.combined_model.evaluate(x=val_data, verbose=verbose)

    def save_model(self):  # , model_weights_path='/work3/s184399/DIRECTORY/model_weights'):
        """
            Saves the model. If the model has not been created, it throws an error.
        """
        assert self.has_model()
        # model_weights_path = model_weights_path.replace('DIRECTORY', self.directory)
        # self.net.save_weights(model_weights_path)
        self.combined_model.save_weights(os.path.join(self.directory, 'model_weights'))

    def save_model_and_training_curves(self, model_weights_path='/work3/s184399/DIRECTORY/model_weights'):
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
        ds = dummy()
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

    def mask_proportion_experiment(self):
        """
            Runs the experiment where we try different (fixed) proportions for dynamic masking.
            Returns the accuracy on the validation set with masked hints.
            p is the proportion of kept words.
        """
        inc = 0.1
        for p in np.arange(0.0, 1 + inc, inc, dtype=np.float32):
            masked_ds = self.dataset.mask_words(p=tf.constant(p),
                                                dataset=self.dataset.validation_set)
            masked_ds = make_batched_and_prefetched(masked_ds)
            self.experiment_results.append(self.combined_model.evaluate(masked_ds, verbose=2))

        return self.experiment_results

    def regularization_rate_experiment(self,
                                       rates,
                                       combination_op='addition',
                                       bert_final_op='pooled_output',
                                       train_kb='wordnet',
                                       val_kb='wordnet',
                                       drop_observations_with_no_hints=True,
                                       train_val_split=0.1,
                                       epochs=4):
        """
            Tries out different regularization rates. Trains for 4 epochs, then validates on the (large) validation set.

            Inputs:
            - rates : A list of floating point numbers specifying the regularization rates.

            Outputs:
            - The accuracies on the validation set.
        """
        self.get_dataset(train_kb=train_kb,
                         val_kb=val_kb,
                         train_val_split=train_val_split,
                         use_reduced_dataset=False,
                         seed=None)  # Make a much smaller dataset for faster training
        if drop_observations_with_no_hints:
            self.dataset.drop_observations_without_hints(for_all_partitions=True)
        for rate in rates:
            self.new_model(regularization_rate=rate,
                           combination_op=combination_op,
                           bert_final_op=bert_final_op)
            self.train_model(epochs=epochs,
                             dynamic_masking=True,
                             dynamic_p=tf.constant(0.7),
                             validation=False)  # Don't do validation here.
            # It'll take forever
            acc = self.evaluate(dynamic_masking=True)  # Slooow: Large validation set.
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

    def bert_experiment(self):
        """
            Experiment description:
                We run the masked inputs through BERT and store the output vectors. We also store the classification.

                The maskings that will be used are p=0.7 and p=0.2 on the validation set.
        """
        bert = get_bert()
        self.get_dataset(train_test_split=0.2)

        # Create the masked datasets in which we discard the images.
        ds1 = make_batched_and_prefetched(
            self.dataset.mask_words(p=tf.constant(0.2),
                                    dataset=self.dataset.training_set).map(lambda X, y: (X['description'],))
        )
        ds2 = make_batched_and_prefetched(
            self.dataset.mask_words(p=tf.constant(0.7),
                                    dataset=self.dataset.training_set).map(lambda X, y: (X['description'],))
        )

        self.experiment_results.append(bert.predict(ds1))
        self.experiment_results.append(bert.predict(ds2))
        self.save_experiment_results()

    def bert_with_dense_experiment(self):
        """
            Experiment description:
                We run the masked inputs through BERT and store the output vectors. We also store the classification.
                The outputs are the BERT embeddings run through the following dense layers

                The maskings that will be used are p=0.7 and p=0.2 on the validation set.
        """
        self.get_dataset(train_test_split=0.2)
        self.combined_model.log_vectors = True

        # Create the masked datasets in which we discard the images.
        ds1 = make_batched_and_prefetched(
            self.dataset.mask_words(p=tf.constant(0.2), dataset=self.dataset.training_set)
        )
        ds2 = make_batched_and_prefetched(
            self.dataset.mask_words(p=tf.constant(0.7), dataset=self.dataset.training_set)
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
        self.dataset.get_dataset(train_test_split=0.2)

        ds = make_batched_and_prefetched(
            self.dataset.training_set.map(lambda X, y: (X['image'],))
        )

        self.experiment_results.append(resnet.predict(ds))
        self.save_experiment_results()

    def resnet50_no_top_experiment(self):
        """
            Here we just log the outputs from the ResNet50. The outputs will be the embeddings.
        """
        resnet = get_resnet50_no_top()
        self.get_dataset(train_val_split=0.8)

        ds = make_batched_and_prefetched(
            self.dataset.training_set.map(lambda X, y: (X['image'],))
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
        val_set = self.dataset.mask_words(dataset=self.dataset.validation_set, seed=42, p=tf.constant(0.7))
        val_no_text = make_batched_and_prefetched(
            val_set.map(lambda X, y: X['image'])
        )
        val_set = make_batched_and_prefetched(val_set)
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

    def learning_curve_experiment(self, *args, **kwargs):
        for i in [0.2, 0.4, 0.6, 0.8]:
            self.new_model(*args, **kwargs)
            print("Train on wikipedia, validate on wordnet. Train: {:.2f}, val: {:.2f}".format(i, 1-i))
            self.dataset = DatasetHandler(
                train_kb='wikipedia',
                val_kb='wordnet',
                train_val_split=i
            ).drop_observations_without_hints(for_all_partitions=True)
            self.train_model(epochs=1, dynamic_masking=True, dynamic_p=tf.constant(0.7), validation=True)
            # Also validate on wikipedia separately
            print("Validate on wikipedia")
            self.dataset = DatasetHandler(
                train_kb='wordnet',
                val_kb='wikipedia',
                train_val_split=i
            ).drop_observations_without_hints(for_all_partitions=True)
            self.evaluate(dynamic_p=tf.constant(0.7), dynamic_masking=True)
            # model.train(x=train, validation_data=val, epochs=1, verbose=2)


if __name__ == '__main__':
    """
    experiment1 = HintsExperiment(
        directory='/work3/s184399/trained_models/weighted_sum_avgseq_trainwnet_valwnet_dynamic_masking_70pct_reg_1e-2'
    )

    print("Weighted sum + avg. seq. Regularization rate = 1e-2.")
    print("Training on WordNet (p_kept=0.7), validating on WordNet. Testing on both individually")
    experiment1.new_model(
        regularization_rate=1e-2,
        combination_op='weighted sum',
        bert_final_op='avg sequence'
    )
    ds_train = DatasetHandler(
        train_kb='wordnet',
        val_kb='wordnet',
        test_kb='wordnet'
    )
    ds_val = DatasetHandler(
        train_kb='wikipedia',
        val_kb='wikipedia',
        test_kb='wikipedia'
    )
    ds_val.drop_observations_without_hints(for_all_partitions=True)
    experiment1.dataset = ds_train
    experiment1.train_model(epochs=2, dynamic_masking=True, dynamic_p=tf.constant(0.7))
    print("Evaluate on test set: WordNet")
    experiment1.evaluate(evaluate_on='test')
    experiment1.dataset = ds_val
    print("Evaluate on test set: Wikipedia")
    experiment1.evaluate(evaluate_on='test')
    experiment1.save_model()"""

    print("Mask proportion experiment for addition-based model with avg. sequence trained on wordnet with p_kept=70pct")
    experiment1 = HintsExperiment(
        directory='/work3/s184399/trained_models/addition_avgseq_trainwnet_valwnet_dynamic_masking_70pct'
    )
    experiment1.new_model(combination_op='addition', bert_final_op='avg sequence')
    experiment1.load_model()
    ds_wn = DatasetHandler(
        train_kb='wordnet',
        val_kb='wordnet',
        train_val_split=0.8
    )#.drop_observations_without_hints(for_all_partitions=True)
    ds_wiki = DatasetHandler(
        train_kb='wordnet',
        val_kb='wikipedia',
        train_val_split=0.8
    ).drop_observations_without_hints(for_all_partitions=True)
    print("Conducting the mask proportion experiment (WordNet)")
    experiment1.dataset = ds_wn
    experiment1.experiment_results = []
    print(experiment1.mask_proportion_experiment())
    print("Conducting the mask proportion experiment (Wikipedia)")
    experiment1.dataset = ds_wiki
    experiment1.experiment_results = []
    print(experiment1.mask_proportion_experiment())

    #experiment2.learning_curve_experiment(combination_op='addition', bert_final_op='avg sequence')

    # experiment.unit_test()

    #print("Regularization experiment. Weighted_sum + avg seq. Trained on wordnet, validated on wordnet.")
    #experiment1.regularization_rate_experiment(
    #    [1e-5, 1e-4, 1e-3, 1e-2],
    #    combination_op='weighted sum',
    #    bert_final_op='avg sequence',
    #    train_kb='wordnet',
    #    val_kb='wordnet',
    #)
    #experiment1.save_experiment_results()

    ## Train model and save weights
    # print("Dataset: Training on Wikipedia, Validating on WordNet")
    # ds = DatasetHandler(train_kb='wikipedia', val_kb='wordnet')
    # ds.drop_observations_without_hints(for_all_partitions=True)
    # experiment1.dataset = ds
    # experiment2.dataset = ds

    # print("Regularization rates experiment")
    """experiment1.regularization_rate_experiment(
        [1e-7, 1e-5, 1e-3, 1e-1, 1e+1],
        train_kb='wikipedia',
        combination_op='weighted sum',
        bert_final_op='pooled_output'
    )
    experiment1.save_experiment_results()"""

    # print("Model: Addition-based, avg. sequence")
    # experiment2.regularization_rate_experiment(
    #    [1e-7, 1e-5, 1e-3, 1e-1, 1e+1],
    #    train_kb='wikipedia',
    #    combination_op='addition',
    #    bert_final_op='avg sequence'
    # )
    # experiment2.save_experiment_results()

    #print("Weighted sum (with sigmoid) - avg sequence. Trained on wikipedia (p=0.7), validated on wordnet")
    #experiment1.get_dataset(train_kb='wikipedia', val_kb='wordnet')
    #experiment1.dataset.drop_observations_without_hints(for_all_partitions=True)
    #experiment1.new_model(combination_op='weighted sum', bert_final_op='avg sequence')
    #experiment1.train_model(epochs=2, dynamic_masking=True, dynamic_p=tf.constant(0.7), validation=True)
    #experiment1.save_model_and_training_curves()
    #experiment1.evaluate(evaluate_on='validation')

    # experiment.get_dataset(val_kb='wikipedia', train_val_split=0.8)
    # experiment.dataset.drop_observations_without_hints()
    # experiment.new_model(combination_op='concatenation')
    # experiment.load_model(model_weights_path='/work3/s184399/trained_models/concatenation_dynamic_masking_70pct/model_weights')
    # experiment.evaluate()

    # print(experiment.mask_proportion_experiment())
    # experiment.save_experiment_results()

    # experiment.bert_experiment()
    # experiment.resnet50_with_top_experiment()

    # experiment.new_model(combination_op='addition')
    # experiment.load_model(
    #    model_weights_path='/work3/s184399/trained_models/addition_dynamic_masking_70pct_regularization_1e-4'
    # )   # TODO: Test the load_model method.  What is this???
    # experiment.correct_bert_experiment()#correction_experiment()
"""
    Get data from HPC:
scp s184399@transfer.gbar.dtu.dk:~/bscproj/hints_experiment/* '/mnt/c/Users/mikke/OneDrive/Dokumenter/DTU documents/7. semester/Bachelor projekt/TF HPC GPU prediction/hints_experiment'

    Transfer code to HPC:
scp * s184399@transfer.gbar.dtu.dk:~/bscproj

"""
