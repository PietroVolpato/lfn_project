LFN PROJECT: Evaluation of Node Embedding Algorithms
Authors: Filippo D'Emilio, Pietro Volpato

Repository Structure
/data
This folder contains the datasets used in the project. Each graph is stored in its own subfolder, with the following files available:
* .edges file: Defines the graph structure.
* Node labels and/or node attributes (if available).
Additionally, the graph_references.txt file provides links to detailed information about each graph, including their source and how to download them.

/result
This folder contains the computed embedding vectors for the nodes in each graph. The naming convention of the files specifies the graph, algorithm, and dimension of the embeddings (e.g., facebook_LINE_128.npy).
* The embeddings are stored as a 2D numpy array, where:
* Each row corresponds to the embedding vector of a node.
* Rows are ordered based on the integer renaming of nodes (from 0 to |V|-1).
* For example, row index 5 corresponds to the embedding vector of the node labeled 5.
Note: If embeddings for the same graph, algorithm, and dimension already exist, they will be overwritten upon recomputation.

/src
This folder contains repositories and scripts for various node embedding algorithms and the notebook produced by us.

Core Notebooks
1) LFN_project_embeddings.ipynb
This notebook is responsible for computing node embeddings.
* Features:
* Select the embedding algorithm and graph.
* Tune algorithm-specific parameters to experiment with different configurations.
* Save computed embeddings to the /result folder.
Important: If an embeddings file already exists for the chosen graph, algorithm, and dimension, it will be overwritten.
2) LFN_project_metrics.ipynb
This notebook implements metrics to evaluate the quality of the embeddings.
* Features:
o Evaluate reconstruction error and neighborhood preservation for all available embeddings.
o Perform tasks on a specific graph-algorithm pair:
* Node classification: Train an SVC classifier using node labels (if available; see graph_references.txt).
* Link prediction: Train a logistic regression classifier to predict the existence of edges.
* Clustering: Use k-means to cluster the embedding space and visualize the clustering on the original graph.

How to Use
1. LFN_project_embeddings.ipynb :
o Run all the setup cells at the beginning
o Chose an algorithm, and run all the cells with its functions definitions. (Tune parameters if you want to)
o Select a graph setting the proper graph key
o Run the cell for the embedding computation (might take a while)
2. LFN_project_metrics.ipynb:
o Run all the setup cells at the beginning (include embedding loading).
o Chose a metric or task and run all the relative function definitions.
o Set up the graph key and embedding key with the pair graph-algorithm you want to evaluate. For reconstruction error and neighbourhood preservation (fast computation) you can define a list of keys, for node classification, link prediction and clustering you have to fix a graph-algorithm pair.
o Visualize and analyze the results.

