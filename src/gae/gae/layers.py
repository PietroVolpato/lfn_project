import tensorflow as tf
from gae.initializations import weight_variable_glorot

# Global dictionary for assigning unique IDs to layers
_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    """Assigns unique IDs to layers."""
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def dropout_sparse(x, keep_prob, num_nonzero_elems):
    """Dropout for sparse tensors."""
    noise_shape = [num_nonzero_elems]
    random_tensor = keep_prob + tf.random.uniform(noise_shape)
    dropout_mask = tf.cast(tf.math.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse.retain(x, dropout_mask)
    return pre_out * (1.0 / keep_prob)


class Layer(tf.keras.layers.Layer):
    """Base class for layers. Defines the basic API for all layers."""
    def __init__(self, name=None, logging=False, **kwargs):
        super(Layer, self).__init__(name=name, **kwargs)
        layer_name = self.__class__.__name__.lower()
        self.name = name or f"{layer_name}_{get_layer_uid(layer_name)}"
        self.logging = logging
        self.issparse = False

    def call(self, inputs, training=None):
        return inputs


class GraphConvolution(Layer):
    """Basic graph convolution layer for undirected graphs without edge labels."""
    def __init__(self, input_dim, output_dim, adj, dropout=0.0, act=tf.nn.relu, **kwargs):
        super(GraphConvolution, self).__init__(**kwargs)
        self.adj = adj
        self.act = act
        self.dropout = dropout
        self.weights = self.add_weight(
            name="weights",
            shape=(input_dim, output_dim),
            initializer=weight_variable_glorot,
            trainable=True,
        )

    def call(self, inputs, training=None):
        x = inputs
        if training:
            x = tf.nn.dropout(x, rate=1-self.dropout)
        x = tf.matmul(x, self.weights)
        x = tf.sparse.sparse_dense_matmul(self.adj, x)
        return self.act(x)


class GraphConvolutionSparse(Layer):
    """Graph convolution layer for sparse inputs."""
    def __init__(self, input_dim, output_dim, adj, features_nonzero, dropout=0.0, act=tf.nn.relu, **kwargs):
        super(GraphConvolutionSparse, self).__init__(**kwargs)
        self.adj = adj
        self.features_nonzero = features_nonzero
        self.act = act
        self.dropout = dropout
        self.issparse = True
        self.weight = self.add_weight(
            name="weights",
            shape=(input_dim, output_dim),
            initializer=tf.keras.initializers.GlorotUniform(),
            trainable=True,
        )

    def call(self, inputs, training=None):
        x = inputs
        if training:
            x = dropout_sparse(x, 1 - self.dropout, self.features_nonzero)
        x = tf.sparse.sparse_dense_matmul(x, self.weight)
        x = tf.sparse.sparse_dense_matmul(self.adj, x)
        return self.act(x)


class InnerProductDecoder(Layer):
    """Decoder layer for link prediction."""
    def __init__(self, input_dim, dropout=0.0, act=tf.nn.sigmoid, **kwargs):
        super(InnerProductDecoder, self).__init__(**kwargs)
        self.dropout = dropout
        self.act = act

    def call(self, inputs, training=None):
        if training:
            inputs = tf.nn.dropout(inputs, rate=1-self.dropout)
        x = tf.matmul(inputs, tf.transpose(inputs))
        return self.act(x)
