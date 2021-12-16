# improving_visual_classification_using_textual_hints
This repository contains the code used for my bachelor thesis at the Technical University of Denmark (DTU)

At the moment, everything is a bit unstructured and most of the documentation in the code is fairly outdated. I uploaded the code in the state it was in when performing the experiments. Since I have had to rename a lot of things after I found better terminology (e.g. in the litterature) than was initially used, there is a discrepancy between some of the terminology in the code and in the report. For now, I here provide a list of the terminology that has changed which can then be used for interpreting the code:
* `avg sequence`, average sequence has changed into *average word embedding*.
* `dynamic_masking` has changed into *dynamic dropping* as this made more sense.
* `dynamic_p` has changed into $p_{\rm kept}$.
* `decreasing_p` could be used for curriculum learning, although I did not have the time to explore this approach.
* `training_set` is the *training split*
* `validation_set` is the *validation split*
* `test_set` is the *validation set*

Lastly, the Wikipedia hints should be provided as a file called `wiki_descriptions.json` in the working directory. They can be obtained by running the file `Wikipedia 2.ipynb`.

**Obtaining ImageNet:**
First download ImageNet2012 from the official website. Then change the directory in `get_imagenet.py`. Then run it to set up ImageNet for first time use with TensorFlow.

## Experiments
### Similarity correlation experiments
In order to run the experiment where I measure the correlation between the semantic similarity and the similarity for the ResNet50-based embeddings, run the following in the shell:
```
usage: experiment.py [-h] [--wordsim [{path,lch,wup}]]
                    [--embedsim [{cosine}]] [--stratified STRATIFIED]
                    {resnet50,vgg19,resnet152v2}

Compute the pairwise similarities of the Neural Network-based embedding 
(being the output of the second last layer) of images from the ImageNet
dataset, and the pairwise similarities of the corresponding labels in the 
WordNet taxonomy.

positional arguments:
  {resnet50,vgg19,resnet152v2}

optional arguments:
  -h, --help            show this help message and exit
  --wordsim [{path,lch,wup}]
  --embedsim [{cosine}]
  --stratified STRATIFIED
```

For the experiment where the correlation is measured between the semantic similarity and the similarity of the BERT-based sentence embeddings, run the file `BERT_and_WordNet.py`.

### BERT-based classifier
For this experiment, run `Bert_classifier_experiment.py`.

### Improving visual classification with textual embeddings
This experiment can be run from the file `image_plus_annotation_experiment.py`. I here give an introduction:

#### Instantiating an experiment and fetching a dataset
In order to create an instance of the experiment class, provide it with a directory in which to save results and models if prompted. Some of the experiments will automatically save, and some won't.

Run the `.get_dataset` method. This takes the arguments: `train_kb`, `val_kb`, `test_kb` (the knowledge base in each partition - can be `wordnet`, `wikipedia` or `both`), `train_val_split` (the training to validation split ratio in hold-out), `use_reduced_dataset` (for testing purposes only) and `seed` (to seed the random generator).

In order to drop the observations without hints (e.g. when using wikipedia for training), call `.dataset.drop_observations_without_hints` which takes an argument `for_all_partitions` (whether to do it across the partitions or only on the individual partitions).

#### Creating, loading and saving a model
In order to load a model, you must first create one in which to load the weights. The architecture must (of course) be the exact same.
A model can be created by calling `.new_model()`and specifying it with parameters such as `combination_op` (`addition`, `concatenation` or `weighted sum`) and `bert_final_op` (`avg sequence` or `pooled_output`).

In order to then load a model, simply call `.load_model()`. If given no path, it uses the one given at instantiation of the experiment class. Otherwise, a weight-path can be specified.

In order to save a model, simply call `.save_model()` and it will be saved to the path given upon instantiation.

#### Training and evaluating a model
To train, run the method `.train_model` which takes the arguments `epochs`, `dynamic_masking` (dynamic dropping, boolean), `dynamic_p` ($p_{\rm kept}$ - a `tf.constant`), `decreasing_p` (for use in curriculum learning) and `validation` (whether to validate on the validation split).

To evaluate a model on the dataset, run `.evaluate` which takes the arguments `dynamic_masking`, `dynamic_p`, `verbose` (whether to print out a line at the end or for every observation) and `evaluate_on` (options: `validation` or `test`).

#### Running the other experiments
In order to run the experiment with masking different proportions of the hints, run `.mask_proportion_experiment()`. This takes no arguments.

In order to run the regularization rate experiment, call `.regularization_rate_experiment` with arguments `rates` (a list of different rates to test), `combination_op`, `bert_final_op` (as for creating a model), `train_kb`, `val_kb`, `drop_observations_with_no_hints`, `train_val_split` (as for getting the dataset), and `epochs`(as for training the models).

In order to run the learning curve experiment, call `.learning_curve_experiment` which takes the same arguments as `.new_model`.
