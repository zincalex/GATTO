import networkx as nx

#region FilePaths
GRAPHS = {
    "EMAIL_EU_CORE" : {
        "graph_name":"email_eu_core",
        "edges_file" : "Data/email_eu_core/email-Eu-core.txt",
        "label_file" : "Data/email_eu_core/email-Eu-core-department-labels.txt",
        "communities" : 42
    },

    "WIKI_TOPCATS" : {
        "graph_name":"wiki_topcats",
        "edges_file" : "Data/wiki_topcats/wiki-topcats.txt",
        "label_file" : "Data/wiki_topcats/wiki-topcats-categories-std.txt",
        "label_file_no_std" : "Data/wiki_topcats/wiki-topcats-categories.txt",
        "communities" : 17364
    },

    "COM_AMAZON" : {
        "graph_name":"com_amazon",
        "edges_file" : "Data/com_amazon/com-amazon.ungraph.txt",
        "label_file" : "Data/com_amazon/com-amazon.all.dedup.cmty-std.txt",
        "label_file_no_std" : "Data/com_amazon/com-amazon.all.dedup.cmty.txt",
        "communities" : 75149
    },

    "CORA_TEST" : {
        "graph_name":"cora",
        "edges_file" : "Data/cora_test/cora.edges",
        "label_file" : "Data/cora_test/cora.node_labels",
        "communities" : 7
    }
}
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
        if tmp[0] != "":
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

if __name__ == "__main__":
    print("STARDARDIZING LABELS FILES")
    convert_com_amazon_to_standard(GRAPHS["COM_AMAZON"]["label_file_no_std"],GRAPHS["COM_AMAZON"]["label_file"])
    print("COM_AMAZON file processed")
    convert_wiki_topcats_to_standard(GRAPHS["WIKI_TOPCATS"]["label_file_no_std"],GRAPHS["WIKI_TOPCATS"]["label_file"])
    print("WIKI_TOPCATS file processed")