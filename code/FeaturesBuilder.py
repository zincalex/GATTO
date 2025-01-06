import GraphBuilder as gBuild
import networkx as nx
from node2vec import Node2Vec
import sklearn.cluster as sk
from joblib import parallel_backend
import pandas as pd
import numpy  as np
import sys
import math
import pickle 

GRAPH_DUMP_FOLDER = "graph_dump/"

# 
# RUN GRAPHBUILDER BEFORE FEATUREBUILDER
# 

def feature_enriching(graph_name: str, num_worker:int, verbose: bool = True):
    """add features to graph nodes

    Parameters
    ----------
    graph_name : str
        could be one between EMAIL_EU_CORE,WIKI_TOPCATS,COM_AMAZON
    num_worker: int 
        number of workers to compute the embedding
    verbose: bool
        set the verbosity of Node2Vec subroutine (defaul: True --> verbosity on)

    The script saves the enanched graph as .pickle format in the folder 'graph_dump/'
    """

    #graph = gBuild.build_data(gBuild.GRAPHS[graph_name]["edges_file"],gBuild.GRAPHS[graph_name]["label_file"])
    graph = pickle.load(open('graph_dump/Cora_graph.pickle', 'rb'))
    graphEmb = Node2Vec(graph,workers=num_worker,quiet=not(verbose)).fit().wv.vectors

    if math.log(gBuild.GRAPHS[graph_name]["communities"],10) < 2:
        with parallel_backend('threading', n_jobs=num_worker):
            clusters = sk.KMeans(n_clusters=gBuild.GRAPHS[graph_name]["communities"],init="k-means++").fit_predict(graphEmb)
    else:
        with parallel_backend('threading', n_jobs=num_worker):
            clusters = sk.AgglomerativeClustering(n_clusters=gBuild.GRAPHS[graph_name]["communities"]).fit_predict(graphEmb)

    predLabels = {i: clusters[i] for i in range(0,clusters.size)}
    nx.set_node_attributes(graph,predLabels,"pred_label")

    #features enriching 
    nx.set_node_attributes(graph,nx.degree_centrality(graph),"dg")
    nx.set_node_attributes(graph,nx.betweenness_centrality(graph),"bv")
    nx.set_node_attributes(graph,nx.closeness_centrality(graph),"cl")
    #nx.set_node_attributes(graph,nx.clustering(graph),"cc")
    if verbose : print("SAVING THE GRAPH ",graph_name,"AS ",gBuild.GRAPHS[graph_name]["graph_name"],".pickle")
    pickle.dump(graph, open(GRAPH_DUMP_FOLDER+gBuild.GRAPHS[graph_name]["graph_name"]+".pickle", "wb"))


def feature_computation(graph_name: str, out_name: str, num_community: int, num_worker: int, verbose: bool = True):
    
    #Graph definition
    print("load pickle")
    graph = pickle.load(open(graph_name, 'rb'))

    #Graph Embedding
    print("compute embeddings")
    graph_embedding = Node2Vec(graph,workers=num_worker,quiet=not(verbose)).fit().wv.vectors
    bound = math.log(num_community,10)

    print("compute clustering")
    if bound < 2:
        with parallel_backend('threading', n_jobs=num_worker):
            clusters = sk.KMeans(n_clusters=num_community,init="k-means++").fit_predict(graph_embedding)
    else:
        with parallel_backend('threading', n_jobs=num_worker):
            clusters = sk.AgglomerativeClustering(n_clusters=num_community).fit_predict(graph_embedding)


    print("compute features")
    dg = pd.DataFrame.from_dict(nx.degree_centrality(graph), orient='index')
    print("Degree centrality: DONE")
    bv = pd.DataFrame.from_dict(nx.betweenness_centrality(graph), orient='index')
    print("Betwenness centrality: DONE")
    cl = pd.DataFrame.from_dict(nx.closeness_centrality(graph), orient='index')
    print("Closenness centrality: DONE")
#    cc = pd.DataFrame.from_dict(nx.clustering(graph), orient="index")
    ia = [None] * len(dg.index)
    for i in range(0, len(dg.index.values)): ia[i] = [dg.index[i]]
    
    cs = [None] * len(dg.index)
    for i in range(0, len(dg.index.values)): cs[i] = [clusters[i]]


    print("merge feature")
    merged_feature = pd.concat([ dg, bv, cl ], axis=1)
    merged_feature = np.hstack((ia, merged_feature))
    merged_feature = np.hstack((merged_feature, cs))
    print(merged_feature)

    if verbose : print("SAVING THE FEATURES MATRIX AS ", out_name)
    pickle.dump(graph, open(out_name, "wb"))

if __name__ == "__main__":
    print("start computing CORA")
    feature_computation("graph_dump/Cora_graph.pickle", "graph_dump/Cora_feature.pickle", 7, 10)
    print("start computing CiteSeer")
    feature_computation("graph_dump/CiteSeer_graph.pickle", "graph_dump/CiteSeer_feature.pickle", 6, 10)
    print("start computing PubMed")
    feature_computation("graph_dump/PubMedDiabetes_graph.pickle", "graph_dump/PubMedDiabetes_feature.pickle", 3, 10)
#    if sys.argv[1] not in gBuild.GRAPHS:
#        print("Wrong graph as input")
#    else:
#        feature_enriching(sys.argv[1], int(sys.argv[2]))

