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
import stellargraph as sg
from stellargraph import StellarGraph
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT
from stellargraph import datasets
from sklearn import model_selection, preprocessing
from tensorflow.keras import Model, optimizers, losses, metrics, layers 

import matplotlib.pyplot as plt
from tqdm.keras import TqdmCallback
import pandas as pd
import networkx as nx
import os
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("-d", type = str, help="Dataset chosen", default = 0)
parser.add_argument("-e", type = int, help="Add extra features", default = 0)
args = parser.parse_args()

if args.d not in ("Cora", "CiteSeer","PubMedDiabetes") :
    raise ValueError(f"Invalid value for -t: {args.d}")
if args.e not in (0, 1) :
    raise ValueError(f"Invalid value for -t: {args.e}. Expected values are 0 or 1.")



class GraphAnalysis:
    def __init__(self):
        self.graph = None
        self.labels = None
    
    def analyze_node_features(self, G):
        # Get the node features as a numpy array
        node_features = G.node_features()
        
        # Convert to pandas DataFrame for better visualization
        feature_df = pd.DataFrame(
            node_features,
            index=G.nodes(),
            columns=[f"feature_{i}" for i in range(node_features.shape[1])]
        )
        
        print("\nNode Features Shape:", node_features.shape)
        print("\nFirst 5 nodes with their features:")
        print(feature_df.head())
        
        # Show non-zero features for the first node
        first_node = feature_df.index[0]
        non_zero_features = feature_df.loc[first_node][feature_df.loc[first_node] != 0]
        
        print(f"\nNon-zero features for paper {first_node}:")
        print(f"This paper contains {len(non_zero_features)} words from the vocabulary")
        print(non_zero_features)
        
        # Count papers containing each word
        word_counts = (feature_df != 0).sum()
        print("\nMost common words in the dataset (by feature number):")
        print(word_counts.nlargest(10))
        
        return feature_df

    def analyze_node_features_nx(self, G_nx): 
        for node, data in G_nx.nodes(data=True):
            print(f"Node {node} attributes: {data}")
            break 

        return
    

    def aggregate_features(self, node_id, node_data) :
        """ Aggregate all features in the node"""
        feature_vector = []

        for key, value in node_data.items():
            #print(f"  {key}: {value}")  # Print feature name and value
            if key != 'label' and key != 'pred_label':
                feature_vector.append(value)
        

        #print(f"Node {node_id} features: {len(feature_vector)}")
        return feature_vector
    

    def load_graph(self, dataset, extra_features_path):
        """Load graph from pickle file and extract node features"""
        # LOAD FEATURES IF NEEDED 
        G, node_subjects = dataset.load()

        if len(extra_features_path) != 0 : 

            with open(extra_features_path, 'rb') as file:
                new_features = pickle.load(file)

        

            node_features = G.node_features()
            updated_features = np.hstack([node_features, new_features])

            # Create pd.DataFrame needed for stellargraph
            node_data = pd.DataFrame(updated_features, index=G.nodes())  # Ensure node index is consistent

            
            edges_list = G.edges()  # List of tuples (start, end)

            # Create DataFrame
            edges_df = pd.DataFrame(edges_list, columns=["source", "target"])
            
            G = StellarGraph(nodes=node_data, edges=edges_df)


        self.graph = G 
        self.labels = node_subjects
        
        


    def train_val_test_data_split(self, train_split, val_split):
        """Split data into train/test sets"""
        
        val_split = val_split / (1 - train_split)
        nodes_indxs = self.labels.index

        # Use sklearn to split the data
        train_nodes, test_nodes = model_selection.train_test_split(self.labels, test_size=1-train_split, stratify=self.labels, random_state=42)
        val_nodes, test_nodes = model_selection.train_test_split(test_nodes, test_size=1-val_split, stratify=test_nodes, random_state=42)
        
        return train_nodes, val_nodes, test_nodes



    
    def train_gat(self, train_nodes, val_nodes, test_nodes, epochs=10000, batch_size=32):
        """Train GAT model"""
        # Nodes: 916, Edges: 13993    26 unique labels  [ 1 21 14  9  4 17 34 11  5 10 36 37  7 22  8 15  3 20 16 38 13  6  0 35 23 19]

        # Conversion to one-hot vectors --- remember here some labels are nomore, hence if problem look here
        target_encoding = preprocessing.LabelBinarizer()
        train_labels_onehot = target_encoding.fit_transform(train_nodes)          # Here we fit because we need to determine the lenght of the encoding
        val_labels_onehot = target_encoding.transform(val_nodes)
        test_labels_onehot = target_encoding.transform(test_nodes)
        

        # The error occurs during the loss computation, where Keras tries to compute the categorical cross-entropy between the true labels and predicted labels.
        print(f"Train labels shape (num nodes, tot labels): {train_labels_onehot.shape}")


        # Create generators
        generator = FullBatchNodeGenerator(self.graph, method="gat", sparse=False)
        
        train_gen = generator.flow(train_nodes.index, train_labels_onehot)
        val_gen = generator.flow(val_nodes.index, val_labels_onehot )
        test_gen = generator.flow(test_nodes.index, test_labels_onehot)
        

        # Create GAT model
        gat = GAT(
            layer_sizes=[8, train_labels_onehot.shape[1]],
            activations=['elu', 'softmax'],
            attn_heads=8,
            generator=generator,
            in_dropout=0.5,
            attn_dropout=0.5,
            normalize=None,
        )
        
        # Get input/output tensors from the GAT model
        x_inp, gat_output = gat.in_out_tensors()

        # Add a Dense layer to map the GAT output to the correct number of classes
        #predictions = layers.Dense(train_labels_onehot.shape[1], activation='softmax')(gat_output)
        #print(f"Model predictions shape: {predictions.shape}")

        # Build and compile the model
        model = Model(inputs=x_inp, outputs=gat_output)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.005),
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
        sg.utils.plot_history(history)
        plt.show()

        # Evaluate
        test_metrics = model.evaluate(test_gen)
        print(f"\nTest Accuracy: {test_metrics[1]:.4f}")
        
        return model, history




def main(): 
    # Disable warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    DATASET = {"Cora" : datasets.Cora(), "CiteSeer" : datasets.CiteSeer(), "PubMedDiabetes" : datasets.PubMedDiabetes()} 
    dataset = DATASET[args.d]

    # PARAMETERS 
    TRAIN_SPLIT = 0.7
    VAL_SPLIT = 0.2
    EPOCHS = 100

    # DATASET   
    extra_features_path = ""
    if args.e == 1 : 
        extra_features_path = "../code/graph_dump/" + args.d + ".pickle"
    


    analyzer = GraphAnalysis()
    analyzer.load_graph(dataset, extra_features_path)
    train_nodes, val_nodes, test_nodes = analyzer.train_val_test_data_split(TRAIN_SPLIT, VAL_SPLIT)
    model, history = analyzer.train_gat(train_nodes, val_nodes, test_nodes, epochs=EPOCHS)

if __name__ == '__main__':
    main()