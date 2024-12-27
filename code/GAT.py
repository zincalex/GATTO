'''
Zincuzzz questo ti serve per la cosa della GAT.
Allora, ti do alcune info:
    - il file .pickle che vedi dentro la cartella 'graph_dump' contiene il grafo con tutte le features:
        - label : la vera label del nodo
        - pred_label: la label predetta da node2vec + clustering
        - dg: grado del nodo
        - bv: betweenness centr del nodo
        - cl: closeness centr del nodo
        - cc: clustering coeff del nodo

        
Per caricare il grafo dal file .pickle ti basta fare 
    G = pickle.load(open('filename.pickle', 'rb'))

Ora se vuoi ottenere le info dal grafo (che è un nx.Graph) ti basta:

ex. label
    label = nx.get_node_attributes(G,"label")
    label[i] #label del nodo num i

Se hai dubbi, chiedimi o leggi la doc ( AI enjoyer :) )
'''

import pickle
import numpy as np
import tensorflow as tf
import networkx as nx
import os
from tensorflow import keras
from tensorflow.keras import layers, Model

class GraphAttentionLayer(layers.Layer):

    # units is number of output features for each node after the attention mechanism
    # kernel_init Specifies the method for initializing weights
    # regula Optional regularization to apply to the weights, which can help prevent overfitting
    def __init__(self, units, kernel_initializer="glorot_uniform", regularizer=None, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
        self.regularizer = tf.keras.regularizers.get(regularizer)

    # Creates the learnable weights for the layer
    # input_shape The shape of the input tensors.
    def build(self, input_shape):
        self.kernel = self.add_weight(
            shape=(input_shape[0][-1], self.units),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.regularizer,
            name="kernel",
        )

        # Used to compute attention scores between node pairs
        #  factor of 2 accounts for concatenating the features of the source and destination nodes during attention calculation
        self.kernel_attention = self.add_weight(
            shape=(self.units * 2, 1),
            trainable=True,
            initializer=self.kernel_initializer,
            regularizer=self.regularizer,
            name="kernel_attention",
        )
        self.built = True

    # forward pass logic of the attention layer
    def call(self, inputs):

        # node_states is the feature vector of all nodes in the graph
        # edges has in each row a pair of node indices defining an edge (source, target)
        node_states, edges = inputs

        # TOKENIZATION & EMBEDDING 
        node_states_transformed = tf.matmul(node_states, self.kernel)

        # ATTENTION SCORES
        node_states_expanded = tf.gather(node_states_transformed, edges)
        node_states_expanded = tf.reshape(node_states_expanded, (tf.shape(edges)[0], -1))
        # compute raw attention scores for each edge, and then activate using a Leaky ReLU function
        attention_scores = tf.nn.leaky_relu(tf.matmul(node_states_expanded, self.kernel_attention))     
        attention_scores = tf.squeeze(attention_scores, -1)                                             

        # NORMALIZE 
        attention_scores = tf.math.exp(tf.clip_by_value(attention_scores, -2, 2)) # TODO CARE ABOUT THIS CLIPPING VALUES
        attention_scores_sum = tf.math.unsorted_segment_sum(
            data=attention_scores,
            segment_ids=edges[:, 0],
            num_segments=tf.reduce_max(edges[:, 0]) + 1,
        )
        attention_scores_sum = tf.repeat(
            attention_scores_sum, tf.math.bincount(tf.cast(edges[:, 0], "int32"))
        )
        attention_scores_norm = attention_scores / attention_scores_sum

        # ADD ----> Aggregate neighbor states
        node_states_neighbors = tf.gather(node_states_transformed, edges[:, 1])
        out = tf.math.unsorted_segment_sum(
            data=node_states_neighbors * attention_scores_norm[:, tf.newaxis],
            segment_ids=edges[:, 0],
            num_segments=tf.shape(node_states)[0],
        )
        return out

# Multi-Head Graph Attention Layer
class MultiHeadGraphAttentionLayer(layers.Layer):
    def __init__(self, units, num_heads=4, merge_type="concat", **kwargs):
        super().__init__(**kwargs)
        self.num_heads = num_heads
        self.merge_type = merge_type
        self.attention_layers = [GraphAttentionLayer(units) for _ in range(num_heads)]

    def call(self, inputs):
        atom_features, pair_indices = inputs
        outputs = [attention_layer([atom_features, pair_indices]) for attention_layer in self.attention_layers]
        if self.merge_type == "concat":
            outputs = tf.concat(outputs, axis=-1)
        else:
            outputs = tf.reduce_mean(tf.stack(outputs, axis=-1), axis=-1)
        return tf.nn.relu(outputs)

# Graph Attention Network Model
class GraphAttentionNetwork(Model):
    def __init__(self, node_states, edges, hidden_units, num_heads, num_layers, output_dim, **kwargs):
        super().__init__(**kwargs)
        self.node_states = node_states
        self.edges = edges
        self.preprocess = layers.Dense(hidden_units * num_heads, activation="relu")
        self.attention_layers = [
            MultiHeadGraphAttentionLayer(hidden_units, num_heads) for _ in range(num_layers)
        ]
        self.output_layer = layers.Dense(output_dim)

    def call(self, inputs):
        node_states, edges = inputs
        x = self.preprocess(node_states)
        for attention_layer in self.attention_layers:
            x = attention_layer([x, edges]) + x
        outputs = self.output_layer(x)
        return outputs

    def train_step(self, data):
        indices, labels = data
        with tf.GradientTape() as tape:
            outputs = self([self.node_states, self.edges])
            loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        grads = tape.gradient(loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))
        return {m.name: m.result() for m in self.metrics}

    def predict_step(self, data):
        indices = data
        outputs = self([self.node_states, self.edges])
        return tf.nn.softmax(tf.gather(outputs, indices))

    def test_step(self, data):
        indices, labels = data
        outputs = self([self.node_states, self.edges])
        loss = self.compiled_loss(labels, tf.gather(outputs, indices))
        self.compiled_metrics.update_state(labels, tf.gather(outputs, indices))
        return {m.name: m.result() for m in self.metrics}


def load_graph(pickle_path):

    # Load the graph from the pickle file
    with open(pickle_path, 'rb') as f:
        G = pickle.load(f)
    
    num_nodes = G.number_of_nodes()
    
    # Extract edge list
    edges = [(u, v) for u, v in G.edges()]

    # Extract node informations
    info = { 'true_label': [], 'pred_label': [], 'dg': [], 'bv': [], 'cl': [], 'cc': []}
    
    for i in range(num_nodes):
        node = G.nodes[i]
        info['true_label'].append(node.get('label', -1))
        info['pred_label'].append(node.get('pred_label', -1))
        info['dg'].append(node.get('dg', -1))
        info['bv'].append(node.get('bv', -1))
        info['cl'].append(node.get('cl', -1))
        info['cc'].append(node.get('cc', -1))
    
    return { 'graph': G, 'edges': edges, 'nodes_information' : info }


def split_data(graph_data, train_split_percentage=0.8):
    
    num_nodes = graph_data['graph'].number_of_nodes()
    node_indices = np.arange(num_nodes)
    nodes_info = graph_data['nodes_information']

    # Shuffle the indices to get a random split
    np.random.shuffle(node_indices)
    
    # Split indices into training and testing sets
    split_idx = int(train_split_percentage * num_nodes)
    train_indices = node_indices[:split_idx]
    test_indices = node_indices[split_idx:]

    # Get the node features and labels for train/test sets
    train_labels = np.array(nodes_info['true_label'])[train_indices]
    test_labels = np.array(nodes_info['true_label'])[test_indices]

    # TODO IN CASE DELETE
    #train_pred_labels = np.array(nodes_info['pred_label'])[train_indices]
    #test_pred_labels = np.array(nodes_info['pred_label'])[test_indices]


    # TODO MAKE IT INDIPENDENT BY THE FEATRUES
    train_node_features = {
        'dg': np.array(nodes_info['dg'])[train_indices],
        'bv': np.array(nodes_info['bv'])[train_indices],
        'cl': np.array(nodes_info['cl'])[train_indices],
        'cc': np.array(nodes_info['cc'])[train_indices], 
        'pred_label' : np.array(nodes_info['pred_label'])[train_indices]
    }

    test_node_features = {
        'dg': np.array(nodes_info['dg'])[test_indices],
        'bv': np.array(nodes_info['bv'])[test_indices],
        'cl': np.array(nodes_info['cl'])[test_indices],
        'cc': np.array(nodes_info['cc'])[test_indices], 
        'pred_label' : np.array(nodes_info['pred_label'])[test_indices]
    }

    return {'train_indices': train_indices, 'test_indices': test_indices,
            'train_labels': train_labels, 'test_labels': test_labels,
            'train_node_features': train_node_features, 'test_node_features': test_node_features,
            'edges': graph_data['edges'] }





def main() : 

    # PARAMETERS 
    TRAIN_SPLIT = 0.8

    # Disable Tensorflow warnings
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

    
    # DATASET   
    dataset_path = "../code/graph_dump/email_eu_core.pickle"
    data = load_graph(dataset_path)

    nodes_info = data['nodes_information']
    unique_labels = list(set(nodes_info['true_label']))             # Values from 0 to 41 
    num_labels = len(unique_labels)
    print(np.array(nodes_info['dg']).shape)
    print(np.array(nodes_info['bv']).shape)
    print(np.array(nodes_info['cl']).shape)
    print(np.array(nodes_info['cc']).shape)
    print(np.array(nodes_info['pred_label']).shape)

    # TODO make it independent
    node_states = np.stack([                                        # Shape: (num_nodes, num_features)  ---> each row a node, colums are features
        np.array(nodes_info['dg']),
        np.array(nodes_info['bv']),
        np.array(nodes_info['cl']),
        np.array(nodes_info['cc']),
        np.array(nodes_info['pred_label'])
    ], axis=-1)

    edges = np.array(data['edges'])

    

    
    # TRAIN TEST SPLIT - 80 20 rule
    data_split = split_data(data, TRAIN_SPLIT)
   


    HIDDEN_UNITS = 100
    NUM_HEADS = 4
    NUM_LAYERS = 2
    OUTPUT_DIM = len(num_labels)

    NUM_EPOCHS = 50
    BATCH_SIZE = 256
    VALIDATION_SPLIT = 0.1
    LEARNING_RATE = 3e-1
    MOMENTUM = 0.9

    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.SGD(LEARNING_RATE, momentum=MOMENTUM)
    accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="acc")
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc", min_delta=1e-5, patience=5, restore_best_weights=True
    )

    # Build model
    gat_model = GraphAttentionNetwork(
        node_states, edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM
    )

    # Compile model
    gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

    gat_model.fit(
        x=data_split['train_indices'],
        y=data_split['train_labels'],
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping],
        verbose=2,
    )

    _, test_accuracy = gat_model.evaluate(x=data_split['test_indices'], y=data_split['test_labels'], verbose=0)

    print("--" * 38 + f"\nTest Accuracy {test_accuracy*100:.1f}%")
    














if __name__ == '__main__':
    main()