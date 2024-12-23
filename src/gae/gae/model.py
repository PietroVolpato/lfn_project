from gae.layers import GraphConvolution, GraphConvolutionSparse, InnerProductDecoder
import tensorflow as tf

class Model(tf.keras.Model):
    def __init__(self, name=None, logging=False):
        super(Model, self).__init__(name=name)
        self.logging = logging
        self.vars = {}

    def build_model(self):
        """Override this method to build the model."""
        raise NotImplementedError

    def call(self, inputs, training=False):
        """Override this method to define the forward pass."""
        raise NotImplementedError

    def fit(self):
        pass

    def predict(self):
        pass

class GCNModelAE(Model):
    def __init__(self, num_features, num_nodes, features_nonzero, hidden1, hidden2, dropout=0.0, **kwargs):
        super(GCNModelAE, self).__init__(**kwargs)
        self.input_dim = num_features
        self.num_nodes = num_nodes
        self.features_nonzero = features_nonzero
        self.hidden1_units = hidden1
        self.hidden2_units = hidden2
        self.dropout_rate = dropout

        # Define layers
        self.hidden1_layer = GraphConvolutionSparse(
            input_dim=num_features,
            output_dim=hidden1,
            adj=None,  # Placeholder, will be set in call method
            features_nonzero=features_nonzero,
            act=tf.nn.relu,
            dropout=dropout,
            logging=self.logging
        )

        self.embedding_layer = GraphConvolution(
            input_dim=hidden1,
            output_dim=hidden2,
            adj=None,  # Placeholder, will be set in call method
            act=lambda x: x,  # Linear activation
            dropout=dropout,
            logging=self.logging
        )

        self.decoder = InnerProductDecoder(
            input_dim=hidden2,
            act=lambda x: x,
            logging=self.logging
        )

    def call(self, inputs, training=False):
        features, adj = inputs
        self.hidden1_layer.adj = adj  # Set adj in call method
        self.embedding_layer.adj = adj  # Set adj in call method
        hidden1 = self.hidden1_layer(features, training=training)
        embeddings = self.embedding_layer(hidden1, training=training)
        reconstructions = self.decoder(embeddings)
        return embeddings, reconstructions

class GCNModelVAE(Model):
    def __init__(self, num_features, num_nodes, features_nonzero, hidden1, hidden2, dropout=0.0, **kwargs):
        super(GCNModelVAE, self).__init__(**kwargs)
        self.input_dim = num_features
        self.num_nodes = num_nodes
        self.features_nonzero = features_nonzero
        self.hidden1_units = hidden1
        self.hidden2_units = hidden2
        self.dropout_rate = dropout

        # Define layers
        self.hidden1_layer = GraphConvolutionSparse(
            input_dim=num_features,
            output_dim=hidden1,
            adj=None,  # Placeholder, will be set in call method
            features_nonzero=features_nonzero,
            act=tf.nn.relu,
            dropout=dropout,
            logging=self.logging
        )

        self.mean_layer = GraphConvolution(
            input_dim=hidden1,
            output_dim=hidden2,
            adj=None,  # Placeholder, will be set in call method
            act=lambda x: x,  # Linear activation
            dropout=dropout,
            logging=self.logging
        )

        self.log_std_layer = GraphConvolution(
            input_dim=hidden1,
            output_dim=hidden2,
            adj=None,  # Placeholder, will be set in call method
            act=lambda x: x,  # Linear activation
            dropout=dropout,
            logging=self.logging
        )

        self.decoder = InnerProductDecoder(
            input_dim=hidden2,
            act=lambda x: x,
            logging=self.logging
        )

    def call(self, inputs, training=False):
        features, adj = inputs
        self.hidden1_layer.adj = adj  # Set adj in call method
        self.mean_layer.adj = adj  # Set adj in call method
        self.log_std_layer.adj = adj  # Set adj in call method
        hidden1 = self.hidden1_layer(features, training=training)
        z_mean = self.mean_layer(hidden1, training=training)
        z_log_std = self.log_std_layer(hidden1, training=training)
        z = z_mean + tf.random.normal([self.num_nodes, self.hidden2_units]) * tf.exp(z_log_std)
        reconstructions = self.decoder(z)
        return z_mean, z_log_std, z, reconstructions
