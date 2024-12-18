import GraphBuilder as gBuild
import networkx as nx
from node2vec import Node2Vec
import sklearn.cluster as sk
import sys
import pickle 

GRAPH_DUMP_FOLDER = "graph_dump/"

# 
# 
# 
# 
# RUN GRAPHBUILDER BEFORE FEATUREBUILDER
# 
# 
# 
# 
# 
# 
# 

# How to load graph from pickle? --> G = pickle.load(open('filename.pickle', 'rb'))

def main(graph_name: str, num_worker:int, verbose: bool = True):
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

    graph = gBuild.build_data(gBuild.GRAPHS[graph_name]["edges_file"],gBuild.GRAPHS[graph_name]["label_file"])
    graphEmb = Node2Vec(graph,workers=num_worker,quiet=not(verbose)).fit().wv.vectors
    clusters = sk.AgglomerativeClustering(n_clusters=gBuild.GRAPHS[graph_name]["communities"]).fit_predict(graphEmb)

    predLabels = {i: clusters[i] for i in range(0,clusters.size)}
    nx.set_node_attributes(graph,predLabels,"pred_labels")

    #features adding 
    nx.set_node_attributes(graph,nx.degree(graph),"dg")
    nx.set_node_attributes(graph,nx.betweenness_centrality(graph),"bv")
    nx.set_node_attributes(graph,nx.closeness_centrality(graph),"cl")
    nx.set_node_attributes(graph,nx.clustering(graph),"cc")
    pickle.dump(graph, open(GRAPH_DUMP_FOLDER+gBuild.GRAPHS[graph_name]["graph_name"]+".pickle", "wb"))
    

if __name__ == "__main__":
    if sys.argv[1] not in gBuild.GRAPHS:
        print("Wrong graph as input")
    else:
        main(sys.argv[1], int(sys.argv[2]))

