# coding=utf-8

import sys
import argparse
import os

import imagenet_interface as inet
import embedding_similarities as embed_sim
import word_similarities as word_sim
import networks as nets
from embedding_similarities import tf
from imagenet_interface import np
"""
    Experiment:
            Compute the pairwise similarities of the Neural Network-based embedding
            (being the output of the second last layer) of images from the ImageNet dataset,
            and the pairwise similarities of the corresponding labels in the WordNet taxonomy.
    
    Info:
        If the specified similarities already exist (the exact same file name as target file), then the
        corresponding similarity will not be recomputed as to decrease computation time. This since the
        images are picked deterministically in order (for now at least).
        
        Also, if the neural network has already had the inputs run through it, we will not even bother with doing it
        again. So if it has been run with e.g. ResNet50 on a GPU once on an e.g. non-stratified dataset, then it can
        be run on a CPU this time, since it is not being repeated (to decrease the resource need).
        
        
    Command-line arguments:
        python experiment.py [-h] [--wordsim [{wordnet_path,wordnet_lch,wordnet_wup,
                                glove.6B.300d_cosine_avg,glove.840B.300d_cosine_avg,spacy_similarity}]] 
                                [--embedsim [{cosine}]] 
                             [--stratified STRATIFIED] {resnet50,vgg19,resnet152v2}
        
        Compute the pairwise similarities of the Neural Network-based embedding (being the output of the second last 
        layer) of images from the ImageNet dataset, and the pairwise similarities of the corresponding labels in the 
        WordNet taxonomy.
        
        positional arguments:
            {resnet50,vgg19,resnet152v2}
        
        optional arguments:
            -h, --help          show help message and exit
            --wordsim [{wordnet_path,wordnet_lch,wordnet_wup,
                                glove.6B.300d_cosine_avg,glove.840B.300d_cosine_avg,spacy_similarity}]
            --embedsim [{cosine}]
            --stratified STRATIFIED   (boolean)
"""


def pred_contains_label(pred_list, true_label):
    """
        To see if the predicted list (e.g. top 5) contains the true label.
    """
    for p in pred_list:
        if p[0] == true_label:
            return True
    return False

# Experiment parameters
NO_OBSERVATIONS = 1000
TOP = 1
STRAT_CLASS_DUPLICATES = 1  # How many class duplicates we want if we stratify the dataset
output_dtype = tf.float32  # Datatype used to store the embeddings and output

###########################
# Parse console arguments #
###########################
parser = argparse.ArgumentParser(
    description="Compute the pairwise similarities of the Neural Network-based embedding " +
                "(being the output of the second last layer) of images from the ImageNet dataset, "
                "and the pairwise similarities of the corresponding labels in the WordNet taxonomy.")
parser.add_argument('network', nargs=1, choices=nets.network_architectures.keys())
parser.add_argument('--wordsim', nargs='?', choices=word_sim.word_similarities.keys())
parser.add_argument('--embedsim', nargs='?', choices=embed_sim.embedding_similarities.keys(), default='cosine')
parser.add_argument('--stratified', type=bool, default=False)

args = parser.parse_args()
net_type_name = args.network[0]
wd_sim_name = None if args.wordsim is None else args.wordsim
em_sim_name = None if args.embedsim is None else args.embedsim
stratified = args.stratified

################################
# Setup the similarity modules #
################################
# If needed, initialize the word_similarity module (i.e. download/prepare the necessary data)
word_sim.init(wd_sim_name)

# Pick the correct image and word similarity functions
wd_sim = word_sim.word_similarities[wd_sim_name]
em_sim = embed_sim.embedding_similarities[em_sim_name]

# Target file names
em_file_name = 'similarities/embedding_similarities ' + net_type_name + ' ' + em_sim_name + ' similarity (first ' + \
               str(NO_OBSERVATIONS) + ' observations ' + ('non-stratified' if not stratified else 'stratified') \
               + ').npy'
wd_file_name = 'similarities/word_similarities ' + wd_sim_name + ' similarity (first ' + str(NO_OBSERVATIONS) + \
               ' observations ' + ('non-stratified' if not stratified else 'stratified') + ').npy'
both_correctly_classified_fname = 'similarities/both_correctly_classified ' + net_type_name + ' (first ' + \
             str(NO_OBSERVATIONS) + ' observations ' + ('non-stratified' if not stratified else 'stratified') + ').npy'

# Check if we need to compute the embedding similarities and word similarities
compute_embedding_sims = not os.path.isfile(os.path.join(os.getcwd(), em_file_name))
compute_word_sims = not os.path.isfile(os.path.join(os.getcwd(), wd_file_name))

#################
# Prepare model #
#################
net_package = None
net = None
preprocess_input = None
decode_predictions = None
embedding_layer_model = None
embedding_dim = 0
if compute_embedding_sims:
    # Pick the correct network and corresponding preprocessing and decoding functions
    net_package = nets.network_architectures[net_type_name]
    net = net_package['net']()
    preprocess_input = net_package['preprocess_input']
    decode_predictions = net_package['decode_predictions']

    # Create the embedding layer model (if needed)
    embedding_layer_model = tf.keras.Model(
        inputs=net.inputs,
        outputs=net.layers[-2].output
    )  # Source: https://keras.io/getting_started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer-feature-extraction
    embedding_dim = embedding_layer_model.output.shape[1]

##########################
# Conduct the experiment #
##########################
dataset = inet.get_dataset_iterator(split='validation')
labels = inet.get_dataset_labels()

true_class = np.zeros(NO_OBSERVATIONS, dtype=np.int32)
class_count = np.zeros(NO_OBSERVATIONS, dtype=np.int32)
label_list = [] # For debugging
was_correctly_classified = np.zeros(NO_OBSERVATIONS, dtype=bool)

embedding_matrix = None
if compute_embedding_sims:
    embedding_matrix = tf.Variable(
        tf.zeros((NO_OBSERVATIONS, embedding_dim), dtype=output_dtype))  # For 2048x50000 datapoints -> 3.27 GB needed!

# Run the images through the network
correct = 0
i = 0  # If not stratified, see i as the incrementing variable in a for loop. If stratified, we just skip replicates.
for sample in dataset:
    if (not stratified and i == NO_OBSERVATIONS) or (stratified and np.all(class_count)):
        break

    label_number = int(sample['label'].numpy())  # Used to obtain the label from the file

    if (not stratified) or (stratified and not class_count[label_number] == STRAT_CLASS_DUPLICATES):
        # Get the label
        label = labels[label_number]
        label_list.append(label)
        true_class[i] = label_number
        class_count[true_class[i]] += 1  # Used to stratify the dataset, if enabled

        if compute_embedding_sims:
            # Prepare image
            img = preprocess_input(tf.image.resize(sample['image'], (224, 224)))
            img = tf.expand_dims(img, axis=0)

            # Predict
            embedding_matrix[i:i + 1, :].assign(tf.cast(embedding_layer_model.predict(img), dtype=output_dtype))
            pred = decode_predictions(net.predict(img), top=TOP)[0]
            if pred_contains_label(pred, label):
                correct += 1
                was_correctly_classified[i] = True

        i += 1

        # Print status
        sys.stdout.flush()
        sys.stdout.write("\rSample: {:d},\tCurrent accuracy (top {:d}): {:.2f}".format(i, TOP, correct / i))

print("\nFinal accuracy (top {:d}): {:f}  ({:d} to {:d})".format(TOP, correct / i, correct, i))
print("Class count: {}".format(class_count))

# Compute the similarities
N_similarities = int(NO_OBSERVATIONS * (NO_OBSERVATIONS + 1) / 2)
embedding_similarities = None if not compute_embedding_sims else tf.Variable(tf.zeros([N_similarities], dtype=output_dtype))
word_similarities = None if not compute_word_sims else tf.Variable(tf.zeros([N_similarities], dtype=output_dtype))
both_correctly_classified = np.zeros(N_similarities, dtype=bool)

print("Computing embedding similarities: {}".format(compute_embedding_sims))
print("Computing word similarities: {}".format(compute_word_sims))
if compute_embedding_sims:
    print("Output files: {}, {}".format(em_file_name, both_correctly_classified_fname))
if compute_word_sims:
    print("Output files: {}".format(wd_file_name))

k = 0
for i in range(NO_OBSERVATIONS):
    for j in range(i, NO_OBSERVATIONS):
        # Image similarities
        if compute_embedding_sims:
            embedding_similarities[k].assign(
                em_sim(embedding_matrix[i:i + 1],
                       embedding_matrix[j:j + 1])
            )
            if was_correctly_classified[i] and was_correctly_classified[j]:
                both_correctly_classified[k] = True
        if compute_word_sims:
            word_similarities[k].assign(
                wd_sim(labels[true_class[i]],
                       labels[true_class[j]])
            )
        k += 1
        if k % 100 == 0:
            print("Similarity progress: {:d}/{:d}".format(k, N_similarities))


if compute_embedding_sims:
    np.save(em_file_name, embedding_similarities.numpy())
    np.save(both_correctly_classified_fname, both_correctly_classified)
if compute_word_sims:
    np.save(wd_file_name, word_similarities.numpy())
