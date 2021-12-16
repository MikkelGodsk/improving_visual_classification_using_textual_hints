import argparse

wn_similarities = {'path': "path", 'lch': "lch", 'wup': "wup"}
in_similarities = {"cosine": "cosine"}
network_architectures = {"resnet50": None, "vgg19": None, "resnet152v2": None}

parser = argparse.ArgumentParser(
    description="Compute the pairwise similarities of the Neural Network-based embedding " +
                "(being the output of the second last layer) of images from the ImageNet dataset, "
                "and the pairwise similarities of the corresponding labels in the WordNet taxonomy.")
parser.add_argument('network', nargs=1, type=str, choices=network_architectures.keys())
parser.add_argument('--wnsim', nargs='?', type=str, choices=wn_similarities.keys())
parser.add_argument('--insim', nargs='?', type=str, choices=in_similarities.keys(), default='cosine')
parser.add_argument('--stratified', type=bool, default=False)

parser.parse_args('resnet50 --insim cosine'.split())