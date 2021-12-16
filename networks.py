import tensorflow.keras.applications.vgg19 as vgg19
import tensorflow.keras.applications.resnet50 as resnet50
import tensorflow.keras.applications.resnet_v2 as resnetv2


"""
    Network architectures stored as a dict. Contains a function to create a network instance, to preprocess input
    and a function to decode the prediction.
    
    The net function is expected to return a network instance.
    The preprocess_input function is expected to return a tensor.
    The decode_prediction function is expected to return a TOP N list of predictions.
"""
network_architectures = {
    'resnet50': {'net': lambda classifier_activation="softmax": resnet50.ResNet50(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000, classifier_activation=classifier_activation),
        'preprocess_input': resnet50.preprocess_input,
        'decode_predictions': resnet50.decode_predictions},
    'vgg19': {'net': lambda: vgg19.VGG19(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000),
        'preprocess_input': vgg19.preprocess_input,
        'decode_predictions': vgg19.decode_predictions},
    'resnet152v2': {'net': lambda: resnetv2.ResNet152V2(
        include_top=True, weights='imagenet', input_tensor=None,
        input_shape=None, pooling=None, classes=1000),
        'preprocess_input': resnetv2.preprocess_input,
        'decode_predictions': resnetv2.decode_predictions},
}