# LFN PROJECT: Evaluation of Node Embedding Algorithms
Authors: Filippo D'Emilio, Pietro Volpato

## Repository Structure
### /data
This folder contains the datasets used in the project. Each graph is stored in its own subfolder, with the following files available:
* .edges file: Defines the graph structure.
* Node labels and/or node attributes (if available).

Additionally, the graph_references.txt file provides links to detailed information about each graph, including their source and how to download them.

### /result
This folder contains the computed embedding vectors for the nodes in each graph. The naming convention of the files specifies the graph, algorithm, and dimension of the embeddings (e.g., facebook_LINE_128.npy). The embeddings are stored as a 2D numpy array, where:

* Each row corresponds to the embedding vector of a node.
* Rows are ordered based on the integer renaming of nodes (from 0 to |V|-1).
* For example, row index 5 corresponds to the embedding vector of the node labeled 5.

Note: If embeddings for the same graph, algorithm, and dimension already exist, they will be overwritten upon recomputation.

### /report_images
Here you cand find images about:
- reconstruction errors of all graphs and algorithms (node2vec, LINE, AW)
- node classification graphs proteins and spam
- link prediction for all graphs and algorithms 
- neighborhood preservation scores of all graphs and algorithms 
- clistering of facebook graph using all algorithms
- some plots representig the results for every algorithm

The file "index.txt" contains information about the content of the images, which are named as integers

### /src
This folder contains repositories and scripts for various node embedding algorithms and the notebook produced by us.

Core Notebooks:

**LFN_project_embeddings.ipynb**

This notebook is responsible for computing node embeddings. The implemented algorithms are: node2vec, LINE, AttentionWalk.
Features:

* Select the embedding algorithm and graph.
* Tune algorithm-specific parameters to experiment with different configurations.
* Save computed embeddings to the /result folder.

Important: If an embeddings file already exists for the chosen graph, algorithm, and dimension, it will be overwritten.

**LFN_project_metrics.ipynb**

This notebook implements metrics to evaluate the quality of the embeddings. Features:

* Evaluate reconstruction error and neighborhood preservation for all available embeddings.

Perform tasks on a specific graph-algorithm pair:
* Node classification: Train an SVC classifier using node labels (if available; see graph_references.txt).
* Link prediction: Train a logistic regression classifier to predict the existence of edges.
* Clustering: Use k-means to cluster the embedding space and visualize the clustering on the original graph.

### How to Use

**LFN_project_embeddings.ipynb :**

* Run all the setup cells at the beginning
* Chose an algorithm, and run all the cells with its functions definitions. (Tune parameters if you want to)
* Select a graph setting the proper graph key
* Run the cell for the embedding computation (might take a while)

**LFN_project_metrics.ipynb:**
* Run all the setup cells at the beginning (include embedding loading).
* Chose a metric or task and run all the relative function definitions.
* Set up the graph key and embedding key with the pair graph-algorithm you want to evaluate. For reconstruction error and neighbourhood preservation (fast computation) you can define a list of keys, for node classification, link prediction and clustering you have to fix a graph-algorithm pair.
* Visualize and analyze the results.

