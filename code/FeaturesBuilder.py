import argparse
import networkx as nx
from node2vec import Node2Vec
import sklearn.cluster as sk
from joblib import parallel_backend
import pandas as pd
import numpy  as np
import math
import pickle 

GRAPH_DUMP_FOLDER = "graph_dump/"
DATASET_LABEL = {"Cora" : 7, "CiteSeer" : 6, "PubMedDiabetes" : 3}  

parser = argparse.ArgumentParser()
parser.add_argument("-d", type = str, help="Dataset chosen", default = "")
parser.add_argument("-p", type = int, help="Set number of workers", default = 1)
parser.add_argument("-v", type = bool, help="Set verbosity", default=False)
args = parser.parse_args()

if args.d not in DATASET_LABEL :
    raise ValueError(f"Invalid value for -d: {args.d}")
if args.p not in range(1, 128) :
    raise ValueError(f"Invalid value for -p: {args.p}. Expected values are between 1 and 128.")
if args.v not in (True,False):
    raise ValueError(f"Invalid value for -v: {args.v}. Expected boolean values (True/False)")



def feature_computation(graph_name: str, out_name: str, num_community: int, num_worker: int, verbose: bool = True):
    
    #Graph definition
    if verbose : print("Loading pickle")
    graph = pickle.load(open(graph_name, 'rb'))

    #Graph Embedding
    if verbose : print("Compute EMBEDINGS...")
    graph_embedding = Node2Vec(graph,workers=num_worker,quiet=not(verbose)).fit().wv.vectors
    bound = math.log(num_community,10)
    if verbose : print("EMBEDDINGS finished")

    if verbose : print("Compute CLUSTERING...")
    if bound < 2:
        with parallel_backend('threading', n_jobs=num_worker):
            clusters = sk.KMeans(n_clusters=num_community,init="k-means++").fit_predict(graph_embedding)
    else:
        with parallel_backend('threading', n_jobs=num_worker):
            clusters = sk.AgglomerativeClustering(n_clusters=num_community).fit_predict(graph_embedding)
    if verbose : print("CLUSTERING finished")

    #Feature Computation
    if verbose : print("Computing features")
    dg = pd.DataFrame.from_dict(nx.degree_centrality(graph), orient='index')
    if verbose : print("Degree centrality: DONE")
    bv = pd.DataFrame.from_dict(nx.betweenness_centrality(graph), orient='index')
    if verbose : print("Betwenness centrality: DONE")
    cl = pd.DataFrame.from_dict(nx.closeness_centrality(graph), orient='index')
    if verbose : print("Closenness centrality: DONE")

    ia = [None] * len(dg.index)
    for i in range(0, len(dg.index.values)): ia[i] = [dg.index[i]]
    
    cs = [None] * len(dg.index)
    for i in range(0, len(dg.index.values)): cs[i] = [clusters[i]]


    if verbose : print("Merging features")
    merged_feature = pd.concat([ dg, bv, cl ], axis=1)
    merged_feature = np.hstack((ia, merged_feature))
    merged_feature = np.hstack((merged_feature, cs))
    if verbose: print(merged_feature)

    if verbose : print("Saving the FEATURES MATRIX as ", out_name)
    pickle.dump(merged_feature, open(out_name, "wb"))

if __name__ == "__main__":
    if args.v : print(f"Start computing {args.d}...")
    feature_computation("graph_dump/"+args.d+"_graph.pickle", "graph_dump/"+args.d+"_feature.pickle", DATASET_LABEL[args.d], args.p, args.v)
    if args.v : print("Computation: DONE")

