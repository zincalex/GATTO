import networkx as nx

#region FilePaths
FILENAME_EDGES_EMAIL_EU_CORE = "Data/email_eu_core/email-Eu-core.txt"
FILENAME_LABELS_EMAIL_EU_CORE = "Data/email_eu_core/email-Eu-core-department-labels.txt"
COMMUNITIES_NUM_EMAIL_EU_CORE = 42

FILENAME_EDGES_WIKI_TOPCATS = "Data/wiki_topcats/wiki-topcats.txt"
FILENAME_LABELS_WIKI_TOPCATS = "Data/wiki_topcats/wiki-topcats-categories.txt"
COMMUNITIES_NUM_WIKI_TOPCATS = 17364

FILENAME_EDGES_COM_AMAZON = "Data/com_amazon/com-amazon.ungraph.txt"
FILENAME_LABELS_COM_AMAZON = "Data/com_amazon/com-amazon.all.dedup.cmty.txt"
COMMUNITIES_NUM_COM_AMAZON = 75149
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
    G: graph (nx.Graph)
        the graph based on edges_file and label_file where each node has community as 'label' attribute
    """

    #region read edges_file:
    internal_edges_file = open(edges_file, "r") 
    edges = []
    for line in internal_edges_file:
        tmp = line.split(" ")
        edges.append((int(tmp[0]), int(tmp[1].strip("\n"))))

    internal_edges_file.close()
    #endregion

    #build Graph
    G = nx.Graph()
    G.add_edges_from(edges)    

    #region read label_file:
    internal_label_file = open(label_file, "r")
    labels = {}
    for line in internal_label_file:
        tmp = line.split(" ")
        labels[int(tmp[0])]=int(tmp[1].strip("\n"))

    internal_label_file.close()
    #endregion

    nx.set_node_attributes(G,labels,"label")

    return G

def convert_com_amazon_to_standard(label_file: str, out_label_file : str):
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