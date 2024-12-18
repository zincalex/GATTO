import GraphBuilder as gBuild
import networkx as nx
from node2vec import Node2Vec
import sklearn.cluster as sk
import sys
import pickle 

GRAPH_DUMP_FOLDER = "graph_dump/"

def saveGraph(graph: nx.Graph,graph_dump_filename : str):
    pickle.dump(graph, open(GRAPH_DUMP_FOLDER+graph_dump_filename+".pickle", "wb"))

# load graph 
# G = pickle.load(open('filename.pickle', 'rb'))

def main(graph: str, num_worker:int):
    emailGraph = gBuild.build_data(gBuild.FILENAME_EDGES_EMAIL_EU_CORE, gBuild.FILENAME_LABELS_EMAIL_EU_CORE)
    emailGraphEmb = Node2Vec(emailGraph,workers=num_worker,quiet=True).fit().wv.vectors
    cluster = sk.AgglomerativeClustering(n_clusters=gBuild.COMMUNITIES_NUM_EMAIL_EU_CORE).fit_predict(emailGraphEmb)

    predLabels = {i: cluster[i] for i in range(0,cluster.size)}
    nx.set_node_attributes(emailGraph,predLabels,"pred_labels")

    #features adding
    nx.set_node_attributes(emailGraph,nx.degree(emailGraph),"dg")
    nx.set_node_attributes(emailGraph,nx.betweenness_centrality(emailGraph),"bv")
    nx.set_node_attributes(emailGraph,nx.closeness_centrality(emailGraph),"cl")
    nx.set_node_attributes(emailGraph,nx.clustering(emailGraph),"cc")
    saveGraph(emailGraph,"email_graph")
    

if __name__ == "__main__":
    main("",int(sys.argv[1]) if  len(sys.argv)==2 else 1)

