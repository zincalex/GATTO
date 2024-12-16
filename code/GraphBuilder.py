import networkx as nx
from node2vec import Node2Vec

#region FilePaths
FILENAME_EDGES_EMAIL_EU_CORE = "Data/email_eu_core/email-Eu-core.txt"
FILENAME_LABELS_EMAIL_EU_CORE = "Data/email_eu_core/email-Eu-core-department-labels.txt"

FILENAME_EDGES_WIKI_TOPCATS = "Data/wiki_topcats/wiki-topcats.txt"
FILENAME_LABELS_WIKI_TOPCATS = "Data/wiki_topcats/wiki-topcats-categories.txt"

FILENAME_EDGES_COM_AMAZON = "Data/com_amazon/com-amazon.ungraph.txt"
FILENAME_LABELS_COM_AMAZON = "Data/com_amazon/com-amazon.all.dedup.cmty.txt"
#endregion

def build_data(edges_file: str, label_file: str):
    """build data from edges file and label file of a dataset

    Parameters
    ----------
    edges_file : str
        the path of file contains edges
    label_file : str
        the path of file contains label to each node

    Returns
    -------
    dict
        dict contains: graph (nx.Graph), labels [ list of tuple(int,int) ] 
    """

    #region read edges_file:
    internal_edges_file = open(edges_file, "r") 
    edges = []
    for line in internal_edges_file:
        tmp = line.split(" ")
        edges.append((int(tmp[0]), int(tmp[1].strip("\n"))))

    internal_edges_file.close()
    #endregion

    #region read label_file:
    internal_label_file = open(label_file, "r")
    labels = []
    for line in internal_label_file:
        tmp = line.split(" ")
        labels.append((int(tmp[0]), int(tmp[1].strip("\n"))))

    internal_label_file.close()
    #endregion

    #build Graph
    G = nx.Graph()
    G.add_edges_from(edges)

    return {
        "graph" : G,
        "labels" : labels
    }

def convert_com_amzon_to_standard(label_file: str, out_label_file : str):
    """convert com-amazon file in a file with our standard format
    Parameters
    ----------
    label_file : str
        the path of file contains label to each node
    out_label_file : str
        the name of the new "standardized"label file
    """

    internal_label_file = open(label_file, "r")
    internal_new_label_file = open(out_label_file, "w")

    row = 0
    for line in internal_label_file:
        tmps = line.split("\t")

        for tmp in tmps:
            internal_new_label_file.write(tmp.strip("\n")+" "+str(row)+"\n")
        row += 1
    internal_label_file.close()
    internal_new_label_file.close()

def convert_wiki_topcats_to_standard(label_file: str, out_label_file : str):
    """convert wiki-topcats file in a file with our standard format

    Parameters
    ----------
    edges_file : str
        the path of file contains edges
    label_file : str
        the path of file contains label to each node
    out_label_file : str
        the name of the new "standardized"label file
    """
    internal_label_file = open(label_file, "r")
    internal_new_label_file = open(out_label_file, "w")

    row = 0
    for line in internal_label_file:
        categoryDivider = line.split("; ")
        tmps = categoryDivider[1].split(" ")

        for tmp in tmps:
            internal_new_label_file.write(tmp.strip("\n")+" "+str(row)+"\n")
        row += 1
    internal_label_file.close()
    internal_new_label_file.close()


"""TEST FOR CLUSTERING
v = build_data(FILENAME_EDGES_EMAIL_EU_CORE, FILENAME_EDGES_EMAIL_EU_CORE)
node2vec = Node2Vec(v["graph"], dimensions=64, walk_length=30, num_walks=200, workers=4)
model = node2vec.fit(window=10, min_count=1, batch_words=4)
"""