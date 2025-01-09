import os
import argparse
import pickle
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
from sklearn import model_selection, preprocessing
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras import Model, optimizers, losses, layers, metrics, callbacks
from tqdm.keras import TqdmCallback

import stellargraph as sg
from stellargraph import StellarGraph, datasets
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT

# User defined libraries
import plots


# PARAMETERS
DATASET = {"Cora" : datasets.Cora(), "CiteSeer" : datasets.CiteSeer(), "PubMedDiabetes" : datasets.PubMedDiabetes()}  
SKIP_TRAINING = False
TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
EPOCHS = 1000
HIDDEN_LAYERS = 8
ATTENTION_HEADS = 6

parser = argparse.ArgumentParser()
parser.add_argument("-d", type = str, help="Dataset chosen", default = 0)
parser.add_argument("-e", type = int, help="Add extra features", default = 0)
args = parser.parse_args()

if args.d not in ("Cora", "CiteSeer","PubMedDiabetes") :
    raise ValueError(f"Invalid value for -d: {args.d}")
if args.e not in (0, 1) :
    raise ValueError(f"Invalid value for -e: {args.e}. Expected values are 0 or 1.")



class GraphAnalysis:
    def __init__(self):
        self.graph = None
        self.labels = None
    

    def load_graph(self, dataset, extra_features_path):
        """
        Loads a graph from stellargraph dataset library and optionally updates it with additional node features.

        Args:
            dataset: The dataset object containing the graph and node subjects.
            extra_features_path (str): Path to a pickle file containing additional node features.
        """
        G, node_subjects = dataset.load()

        # EXTRA FEATURES 
        if len(extra_features_path) != 0 : 

            with open(extra_features_path, 'rb') as file:
                new_features = pickle.load(file)

            # Extract existing node features from the graph
            node_features = G.node_features()

            # Combine existing features with the new features horizontally
            updated_features = np.hstack([node_features, new_features])

            # Create a DataFrame for the updated node features
            node_data = pd.DataFrame(updated_features, index=G.nodes())  # Ensure node index is consistent

            edges_list = G.edges()  # List of tuples (start, end)
            edges_df = pd.DataFrame(edges_list, columns=["source", "target"])
            
            # Create a new StellarGraph object
            G = StellarGraph(nodes=node_data, edges=edges_df)

        self.graph = G 
        self.labels = node_subjects


    def train_val_test_data_split(self, train_split, val_split):
        """
        Splits the graph data into training, validation, and test sets.

        Args:
            train_split (float): Proportion of data to use for training.
            val_split (float): Proportion of the remaining data to use for validation.

        Returns:
            tuple: Contains three DataFrames for training, validation, and test nodes.
        """
        
        val_split = val_split / (1 - train_split)

        # Use sklearn to split the data
        train_nodes, test_nodes = model_selection.train_test_split(self.labels, test_size=1-train_split, stratify=self.labels, random_state=42)
        val_nodes, test_nodes = model_selection.train_test_split(test_nodes, test_size=1-val_split, stratify=test_nodes, random_state=42)
        
        return train_nodes, val_nodes, test_nodes
    

    def train_gat(self, train_nodes, val_nodes, epochs, hidden_layers, attention_heads):
        """
        Trains a Graph Attention Network (GAT) model.

        Args:
            train_nodes: Training nodes and their labels.
            val_nodes: Validation nodes and their labels.
            epochs (int): Number of training epochs.
            hidden_layers (int): Number of hidden layers in the GAT model.
            attention_heads (int): Number of attention heads in the GAT model.

        Returns:
            tuple: Contains the trained GAT model and its training history.
        """


        # Conversion to one-hot vectors 
        target_encoding = preprocessing.LabelBinarizer()
        train_labels_onehot = target_encoding.fit_transform(train_nodes)          # Here we fit because we need to determine the lenght of the encoding
        val_labels_onehot = target_encoding.transform(val_nodes)
    
        # Create generators for full-batch training ---> typically used from GAT
        generator = FullBatchNodeGenerator(self.graph, method="gat", sparse=False)
        train_gen = generator.flow(train_nodes.index, train_labels_onehot)
        val_gen = generator.flow(val_nodes.index, val_labels_onehot )
       
        # Create GAT model
        gat = GAT(
            layer_sizes=[hidden_layers, train_labels_onehot.shape[1]],
            activations=['elu', 'softmax'],
            attn_heads=attention_heads,
            generator=generator,
            in_dropout=0.5,
            attn_dropout=0.5,
            normalize=None,
        )
        
        # Get input/output tensors from the GAT model
        x_inp, gat_output = gat.in_out_tensors()

        # Build and compile the model
        model = Model(inputs=x_inp, outputs=gat_output)
        model.compile(
            optimizer=optimizers.Adam(learning_rate=0.001),
            loss=losses.categorical_crossentropy,
            metrics=['acc']
        )
        
        # Initialize Callbakcs
        if not os.path.isdir("logs"):
            os.makedirs("logs")
        mc_callback = callbacks.ModelCheckpoint("logs/best_model.h5", monitor="val_acc", save_best_only=True, save_weights_only=True)
        tqdm_callback = TqdmCallback()

        # TRAINING
        history = model.fit(
            train_gen,
            epochs=epochs,
            validation_data=val_gen,
            verbose=0, 
            shuffle=False,
            callbacks = [tqdm_callback, mc_callback]
        )
        
        plots.plot_and_save_training_history(history, f"logs/{args.d}_training_history.png")

        return model, history


    def test_gat(self, model, test_nodes, graph_labels) : 
        """
        Tests the trained GAT model on the test nodes.

        Args:
            model: The trained GAT model.
            test_nodes: Test nodes and their labels.
            graph_labels: All labels in the dataset for inverse transformation.

        Returns:
            tuple: Contains predicted labels and true labels for the test nodes.
        """
        # Labels one hot encoding
        target_encoding = preprocessing.LabelBinarizer()
        test_labels_onehot = target_encoding.fit_transform(test_nodes)

        # Prediction
        generator = FullBatchNodeGenerator(self.graph, method="gat", sparse=False)
        test_gen = generator.flow(test_nodes.index, test_labels_onehot)
        y_pred = model.predict(test_gen)

        # Output conversion
        y_pred = y_pred.squeeze()
        y_pred_classes = np.argmax(y_pred, axis=1)
        label_encoder = preprocessing.LabelEncoder()
        label_encoder.fit(graph_labels)  # Fit on all labels in the dataset
        y_pred_labels = label_encoder.inverse_transform(y_pred_classes)

        y_true_labels = test_nodes.values

        return y_pred_labels, y_true_labels



def main(): 
    # Disable warnings
    os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
    dataset = DATASET[args.d]

    # DATASET   
    extra_features_path = ""
    if args.e == 1 : 
        extra_features_path = "../code/graph_dump/" + args.d + ".pickle"
    
    analyzer = GraphAnalysis()
    analyzer.load_graph(dataset, extra_features_path)
    train_nodes, val_nodes, test_nodes = analyzer.train_val_test_data_split(TRAIN_SPLIT, VAL_SPLIT)
    model, history = analyzer.train_gat(train_nodes, val_nodes, EPOCHS, HIDDEN_LAYERS, ATTENTION_HEADS)
    y_pred_labels, y_true_labels = analyzer.test_gat(model, test_nodes, analyzer.labels)

    # Calculate metrics
    accuracy = accuracy_score(y_true_labels, y_pred_labels)
    precision = precision_score(y_true_labels, y_pred_labels, average='weighted')
    recall = recall_score(y_true_labels, y_pred_labels, average='weighted')
    f1 = f1_score(y_true_labels, y_pred_labels, average='weighted')

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")

    # Plot and save confusion matrix
    class_names = np.unique(analyzer.labels)  # Get unique class names
    plots.plot_and_save_confusion_matrix(y_true_labels, y_pred_labels, class_names, save_path=f"logs/{args.d}_confusion_matrix.png")



if __name__ == '__main__':
    main()