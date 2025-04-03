import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """
    Builds a neural network using the Keras library.

    Args:
        nx: number of input features to the network
        layers: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        lambtha: L2 regularization parameter
        keep_prob: probability that a node will be kept for dropout

    Returns:
        The Keras model
    """
    model = K.Sequential()
    regularizer = K.regularizers.L2(lambtha)

    for i in range(len(layers)):
        if i == 0:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=regularizer,
                                     input_shape=(nx,)))
        else:
            model.add(K.layers.Dense(layers[i], activation=activations[i],
                                     kernel_regularizer=regularizer))
        if i < len(layers) - 1:  # Add dropout for all layers except the last
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
