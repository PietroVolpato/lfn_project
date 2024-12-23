import time
import os
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
from sklearn.metrics import roc_auc_score, average_precision_score

from gae.optimizer import OptimizerAE, OptimizerVAE
from gae.input_data import load_data
from gae.model import GCNModelAE, GCNModelVAE
from gae.preprocessing import preprocess_graph, construct_feed_dict, sparse_to_tuple, mask_test_edges

# Settings
class Config:
    learning_rate = 0.01
    epochs = 200
    hidden1 = 32
    hidden2 = 16
    weight_decay = 0.0
    dropout = 0.0
    model = 'gcn_ae'
    dataset = 'cora'
    features = 1  # Use features (1) or not (0)

config = Config()

# Enable GPU
os.environ['CUDA_VISIBLE_DEVICES'] = "0"  # Adjust if you have multiple GPUs

# Load data
adj, features = load_data(config.dataset)

# Store original adjacency matrix (without diagonal entries) for later
adj_orig = adj - sp.dia_matrix((adj.diagonal()[np.newaxis, :], [0]), shape=adj.shape)
adj_orig.eliminate_zeros()

adj_train, train_edges, val_edges, val_edges_false, test_edges, test_edges_false = mask_test_edges(adj)
adj = adj_train

if not config.features:
    features = sp.identity(features.shape[0])  # featureless

# Some preprocessing
adj_norm = preprocess_graph(adj)

# Convert sparse matrices to TensorFlow SparseTensor
features = sparse_to_tuple(features.tocoo())
adj_label = sparse_to_tuple(adj_train + sp.eye(adj_train.shape[0]))

# Model initialization
num_nodes = adj.shape[0]
num_features = features[2][1]
features_nonzero = features[1].shape[0]

if config.model == 'gcn_ae':
    model = GCNModelAE(num_features, num_nodes, features_nonzero, config.hidden1, config.hidden2, config.dropout)
elif config.model == 'gcn_vae':
    model = GCNModelVAE(num_features, num_nodes, features_nonzero, config.hidden1, config.hidden2, config.dropout)

# Optimizer
pos_weight = float(adj.shape[0] * adj.shape[0] - adj.sum()) / adj.sum()
norm = adj.shape[0] * adj.shape[0] / float((adj.shape[0] * adj.shape[0] - adj.sum()) * 2)

optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

# Training loop
@tf.function
def train_step(features, adj_norm, adj_label, dropout):
    with tf.GradientTape() as tape:
        reconstructions = model(features, adj_norm, dropout)
        loss = model.loss(reconstructions, adj_label, pos_weight, norm)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss

# Training
print("Training...")
for epoch in range(config.epochs):
    t = time.time()
    loss = train_step(features, adj_norm, adj_label, config.dropout)

    # Evaluate
    if epoch % 10 == 0 or epoch == config.epochs - 1:
        roc_score, ap_score = model.evaluate(adj_orig, val_edges, val_edges_false)
        print(f"Epoch {epoch+1}, Loss: {loss:.5f}, ROC: {roc_score:.5f}, AP: {ap_score:.5f}, Time: {time.time() - t:.5f}")

# Testing
print("Optimization Finished!")
roc_score, ap_score = model.evaluate(adj_orig, test_edges, test_edges_false)
print(f"Test ROC score: {roc_score:.5f}")
print(f"Test AP score: {ap_score:.5f}")
