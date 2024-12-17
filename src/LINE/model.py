from keras.layers import Embedding, Reshape, Dot, Input
from keras.models import Model
from keras.models import Sequential, Model


def create_model(numNodes, factors):
    # Define inputs for two nodes in a pair
    left_input = Input(shape=(1,))
    right_input = Input(shape=(1,))

    # Shared embedding layer
    embedding_layer = Embedding(input_dim=numNodes + 1, output_dim=factors, input_length=1, mask_zero=False)

    # Embedding outputs
    left_embed = embedding_layer(left_input)
    left_embed = Reshape((factors,))(left_embed)

    right_embed = embedding_layer(right_input)
    right_embed = Reshape((factors,))(right_embed)

    # Dot product to compute similarity between embeddings
    left_right_dot = Dot(axes=1)([left_embed, right_embed])

    # Define models
    model = Model(inputs=[left_input, right_input], outputs=left_right_dot)
    embed_generator = Model(inputs=[left_input], outputs=left_embed)

    return model, embed_generator