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
from tqdm.keras import TqdmCallback
import tensorflow.keras.backend as K

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT

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





def main(): 
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
    NUM_HEADS = 8
    NUM_LAYERS = 2
    OUTPUT_DIM = num_labels

    NUM_EPOCHS = 200
    BATCH_SIZE = 256
    VALIDATION_SPLIT = 0.1
    LEARNING_RATE = 1e-2

    # Using Adam optimizer for better stability
    loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
    optimizer = keras.optimizers.Adam(learning_rate=LEARNING_RATE, clipnorm=1.0)
    accuracy_fn = keras.metrics.SparseCategoricalAccuracy(name="acc")


    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_acc",
        min_delta=1e-5,
        patience=NUM_EPOCHS,  # Set to number of epochs to prevent early stopping
        restore_best_weights=True
    )

    # Build and compile model
    gat_model = GraphAttentionNetwork(
        node_states, edges, HIDDEN_UNITS, NUM_HEADS, NUM_LAYERS, OUTPUT_DIM
    )

    # Compile model
    gat_model.compile(loss=loss_fn, optimizer=optimizer, metrics=[accuracy_fn])

    # Training with a progress bar
    gat_model.fit(
        x=data_split['train_indices'],
        y=data_split['train_labels'],
        validation_split=VALIDATION_SPLIT,
        batch_size=BATCH_SIZE,
        epochs=NUM_EPOCHS,
        callbacks=[early_stopping, TqdmCallback(verbose=1)],  # Added Tqdm progress bar
        verbose=0,  # Disable default logging to use Tqdm
    )

    # Evaluate model
    _, test_accuracy = gat_model.evaluate(x=data_split['test_indices'], y=data_split['test_labels'], verbose=0)
    print(f"Test Accuracy: {test_accuracy*100:.1f}%")

if __name__ == '__main__':
    main()














if __name__ == '__main__':
    main()