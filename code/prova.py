import pickle
import networkx

graph = pickle.load(open("graph_dump/email_eu_core.pickle", 'rb'))
print(type(graph))
print(graph.nodes()[0])