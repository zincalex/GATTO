import networkx as nx
import pandas as pd
import os

import stellargraph as sg
from stellargraph.mapper import FullBatchNodeGenerator
from stellargraph.layer import GAT
from stellargraph import StellarGraph

from tensorflow.keras import layers, optimizers, losses, metrics, Model
from sklearn import preprocessing, feature_extraction, model_selection
from stellargraph import datasets
from IPython.display import display, HTML
import matplotlib.pyplot as plt

def analyze_node_features(G):
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

def analyze_node_features_nx(G_nx): 
        for node, data in G_nx.nodes(data=True):
            print(f"Node {node} attributes: {data}")
            

        return

def main(): 
    dataset = datasets.Cora()
    display(HTML(dataset.description))
    G, node_subjects = dataset.load()

    print("\n=== Node Feature Analysis ===")
    feature_df = analyze_node_features(G)
    G = StellarGraph.to_networkx(G)
    analyze_node_features_nx(G)

    train_subjects, test_subjects = model_selection.train_test_split(
    node_subjects, train_size=140, test_size=None, stratify=node_subjects
    )
    val_subjects, test_subjects = model_selection.train_test_split(
        test_subjects, train_size=500, test_size=None, stratify=test_subjects
    )

    target_encoding = preprocessing.LabelBinarizer()

    train_targets = target_encoding.fit_transform(train_subjects)
    val_targets = target_encoding.transform(val_subjects)
    test_targets = target_encoding.transform(test_subjects)

    generator = FullBatchNodeGenerator(G, method="gat")
    train_gen = generator.flow(train_subjects.index, train_targets)
    gat = GAT(
        layer_sizes=[8, train_targets.shape[1]],
        activations=["elu", "softmax"],
        attn_heads=8,
        generator=generator,
        in_dropout=0.5,
        attn_dropout=0.5,
        normalize=None,
    )
    x_inp, predictions = gat.in_out_tensors()
    model = Model(inputs=x_inp, outputs=predictions)
    model.compile(
        optimizer=optimizers.Adam(lr=0.005),
        loss=losses.categorical_crossentropy,
        metrics=["acc"],
    )
    val_gen = generator.flow(val_subjects.index, val_targets)


    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
    if not os.path.isdir("logs"):
        os.makedirs("logs")
    es_callback = EarlyStopping(
        monitor="val_acc", patience=20
    )  # patience is the number of epochs to wait before early stopping in case of no further improvement
    mc_callback = ModelCheckpoint(
        "logs/best_model.h5", monitor="val_acc", save_best_only=True, save_weights_only=True
    )

    history = model.fit(
        train_gen,
        epochs=50,
        validation_data=val_gen,
        verbose=2,
        shuffle=False,  # this should be False, since shuffling data means shuffling the whole graph
        callbacks=[es_callback, mc_callback],
    )

    sg.utils.plot_history(history)
    test_gen = generator.flow(test_subjects.index, test_targets)
    test_metrics = model.evaluate(test_gen)
    print("\nTest Set Metrics:")
    for name, val in zip(model.metrics_names, test_metrics):
        print("\t{}: {:0.4f}".format(name, val))


if __name__ == '__main__':
    main()