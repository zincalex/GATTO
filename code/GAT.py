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

from gc import callbacks
import pickle
import numpy as np
from stellargraph import StellarGraph
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT
from sklearn import model_selection, preprocessing
from tensorflow.keras import Model, optimizers, losses, metrics, layers 
import test
from tqdm.keras import TqdmCallback
import pandas as pd
import networkx as nx
import os

class GraphAnalysis:
    def __init__(self):
        self.graph = None
        self.labels = None
        
    def aggregate_features(self, node_id, node_data) :
        """ Aggregate all features in the node"""
        feature_vector = []

        for key, value in node_data.items():
            if key != 'label':
                feature_vector.append(value)
        
        return feature_vector
    

    def load_graph(self, pickle_path):
        """Load graph from pickle file and extract node features"""
        with open(pickle_path, 'rb') as f:
            nx_graph = pickle.load(f)
        
        # Extract some information from the graph
        labels = []
        node_ids = []
        for node_id, node_data in nx_graph.nodes(data=True):
            labels.append(node_data['label'])
            node_ids.append(node_id)
            node_data["features_aggregation"] = self.aggregate_features(node_id, node_data)

        # Encode labels
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(labels)
        encoded_labels = pd.Series(label_encoder.transform(labels), index=node_ids)

        # Ensure a sufficient number of labels per node
        label_counts = encoded_labels.value_counts()
        valid_labels = label_counts[label_counts > 10].index
        valid_nodes = [node for node in encoded_labels.index if encoded_labels[node] in valid_labels]

        # Remove invalid nodes from the graph and corresponding data
        nx_graph.remove_nodes_from([node for node in nx_graph.nodes if node not in valid_nodes])
        encoded_labels = encoded_labels.loc[valid_nodes]

        #self.nodes_features = feature_df
        self.labels = encoded_labels

        # Try to convert to StellarGraph
        self.graph = StellarGraph.from_networkx(nx_graph, node_features="features_aggregation",  node_type_attr=None)



    def train_val_test_data_split(self, train_split, val_split):
        """Split data into train/test sets"""
        
        val_split = val_split / (1 - train_split)
        nodes_indxs = self.labels.index

        # Use sklearn to split the data
        train_nodes, test_nodes = model_selection.train_test_split(nodes_indxs, test_size=1-train_split, stratify=self.labels[nodes_indxs], random_state=42)
        val_nodes, test_nodes = model_selection.train_test_split(test_nodes, test_size=1-val_split, stratify=self.labels[test_nodes], random_state=42)
        
        return train_nodes, val_nodes, test_nodes



    
    def train_gat(self, train_nodes, val_nodes, test_nodes, epochs=1000, batch_size=64):
        """Train GAT model"""
        # Nodes: 916, Edges: 13993    26 unique labels  [ 1 21 14  9  4 17 34 11  5 10 36 37  7 22  8 15  3 20 16 38 13  6  0 35 23 19]

        # Conversion to one-hot vectors --- remember here some labels are nomore, hence if problem look here
        target_encoding = preprocessing.LabelBinarizer()
        target_encoding.fit(self.labels)
        train_labels_onehot = target_encoding.transform(self.labels[train_nodes])          # Here we fit because we need to determine the lenght of the encoding
        val_labels_onehot = target_encoding.transform(self.labels[val_nodes])
        test_labels_onehot = target_encoding.transform(self.labels[test_nodes])
        

        # The error occurs during the loss computation, where Keras tries to compute the categorical cross-entropy between the true labels and predicted labels.
        print(f"TR shape : {train_nodes.shape}")
        print(train_labels_onehot.shape[1])
        print(f"Train labels shape: {train_labels_onehot.shape}")
        print(f"Validation labels shape: {val_labels_onehot.shape}")
        print(f"Test labels shape: {test_labels_onehot.shape}")


        # Create generators
        generator = FullBatchNodeGenerator(self.graph, method="gat", sparse=False)
        
        train_gen = generator.flow(train_nodes, train_labels_onehot)
        val_gen = generator.flow(val_nodes, val_labels_onehot )
        test_gen = generator.flow(test_nodes, test_labels_onehot)
        

        # Create GAT model
        gat = GAT(
            layer_sizes=[8, 8],
            activations=['elu', 'softmax'],
            attn_heads=1,
            generator=generator,
            in_dropout=0.3,
            attn_dropout=0.6,
            normalize=None,
        )
        
        # Get input/output tensors from the GAT model
        x_inp, gat_output = gat.in_out_tensors()

        # Add a Dense layer to map the GAT output to the correct number of classes
        predictions = layers.Dense(train_labels_onehot.shape[1], activation='softmax')(gat_output)
        print(f"Model predictions shape: {predictions.shape}")

        # Build and compile the model
        model = Model(inputs=x_inp, outputs=predictions)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.01),
            loss=losses.categorical_crossentropy,
            metrics=['acc']
        )
        
        # Train
        tqdm_callback = TqdmCallback()
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            verbose=0, 
            shuffle=False,
            callbacks = [tqdm_callback]
        )
    
        # Evaluate
        test_metrics = model.evaluate(test_gen)
        print(f"\nTest Accuracy: {test_metrics[1]:.4f}")
        
        return model, history




def main(): 
    # Disable warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


    # PARAMETERS 
    TRAIN_SPLIT = 0.8
    VAL_SPLIT = 0.1 

    

    # DATASET   
    dataset_path = "../code/graph_dump/email_eu_core.pickle"


    analyzer = GraphAnalysis()
    analyzer.load_graph(dataset_path)
    train_nodes, val_nodes, test_nodes = analyzer.train_val_test_data_split(TRAIN_SPLIT, VAL_SPLIT)
    model, history = analyzer.train_gat(train_nodes, val_nodes, test_nodes)

if __name__ == '__main__':
    main()