from node2vec import Node2Vec
import networkx as nx
import numpy as np
import re
import os

def load_graph(path):
    """
    For files with extension .edges
    nodes are renamed as integers, starting from 0
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    G = nx.Graph()
    with open(path, 'rt') as f:
        for line in f:
            if line.startswith('%'):  # Skip comment lines
                continue
            # Split the line based on spaces or commas
            data = re.split(r'[,\s]+', line.strip())
            if len(data) < 2:  # Skip lines that don't have at least two columns
                continue
            # Extract the first two columns (nodes)
            node1, node2 = int(data[0]), int(data[1])
            G.add_edge(node1, node2)
    mapping = {node : i for i,node in enumerate(G.nodes)} # mappoing original : relabeled
    G = nx.relabel_nodes(G, mapping)
    return G

def get_node2vec_embeddings(G, dimensions=128, walk_length=50, num_walks=40, neighborhood_size = 10, p=0.5, q=2):
    """
    Generate node embeddings for a graph using the Node2Vec algorithm.

    Parameters:
        G (networkx.Graph):The input graph for which embeddings are to be generated.
            The graph should have nodes labeled as integers, ideally sequentially starting from 0.
        dimensions (int, optional): The dimensionality of the embedding space. Default is 128.
        walk_length (int, optional): The length of each random walk. Default is 10.
        num_walks (int, optional): The number of random walks to start from each node. Default is 20.
        p (float, optional):
            The return parameter, controlling the likelihood of immediately revisiting a node in the walk.
            A higher value makes it more likely to backtrack. Default is 1.
        q (float, optional):
            The in-out parameter, controlling the likelihood of exploring outward from the starting node.
            A higher value makes it more likely to move outward. Default is 1.
        workers (int, optional): The number of parallel workers for random walk generation and model training. Default is 1.

    Returns:
        np.ndarray: A NumPy array where each row represents the embedding of a node.
            The row index corresponds to the node ID, and each row has `dimensions` elements.
    """
    # Initialize Node2Vec model
    node2vec = Node2Vec(G, dimensions=dimensions, walk_length=walk_length, num_walks=num_walks, p=p, q=q, workers=1)

    # Fit the Node2Vec model and generate embeddings
    model = node2vec.fit(window=neighborhood_size, min_count=1, batch_words=4)

    # Convert embeddings to a NumPy array
    num_nodes = G.number_of_nodes()
    embeddings = np.zeros((num_nodes, dimensions))  # Preallocate array
    for node in G.nodes:
        embeddings[node] = model.wv[str(node)]  # in the vocabulary node names are converted always to strings
    return embeddings

def save(emb, graph_key, embedding_key, emb_dim):
    path = f"../result/embeddings_{graph_key}_{embedding_key}_{emb_dim}.npy"
    np.save(path, emb)
    print(f"Successfully saved the embeddings in {path}")


def main():
    citation_path = '../data/citation/cit-HepTh.edges'
    try:
        G = load_graph(citation_path)
        embeddings = get_node2vec_embeddings(G, dimensions=128, walk_length=80, num_walks=18, neighborhood_size = 16, p=0.25, q=0.25)
        save(embeddings, 'citation', 'node2vec', 128)
    except FileNotFoundError as e:
        print(e)

if __name__ == '__main__':
    main()