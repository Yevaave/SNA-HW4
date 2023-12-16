import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt
import itertools
import nltk
from nltk.tokenize import word_tokenize
from nltk.probability import FreqDist
from collections import Counter
from infomap import Infomap
from operator import itemgetter


data = pd.read_csv('./connections.txt',delimiter=' ',names = ['from','to'])


#1
graph = nx.DiGraph(edges)
graph.add_edges_from(data.to_numpy())

#2
bridges =list(nx.bridges(graph.to_undirected()))
merged = list(itertools.chain(*bridges))
bridge_nodes = set(merged)
print(bridge_nodes)

#3
graph_density = nx.density(graph)
print("Graph Density:", graph_density)

#4
node_degrees = graph.degree()
max_degree_node = max(node_degrees, key=lambda x: x[1])
min_degree_node = min(node_degrees, key=lambda x: x[1])

print("Node with the highest degree:", max_degree_node)
print("Node with the lowest degree:", min_degree_node)

#5
in_degrees = graph.in_degree()
out_degrees = graph.out_degree()

max_in_degree_node = max(in_degrees, key=lambda x: x[1])
max_out_degree_node = max(out_degrees, key=lambda x: x[1])

print("Node with the highest incoming connections:", max_in_degree_node)
print("Node with the highest outgoing connections:", max_out_degree_node)

#6
betweenness = nx.centrality.betweenness_centrality(graph)
highest_betweenness_node = max(graph.nodes, key=betweenness.get)
b=betweenness[highest_betweenness_node]
print("Highest Betweenness Node: {} {}Betweenness: {}".format(highest_betweenness_node,'\n',b ,'\n'))

closeness = nx.centrality.closeness_centrality(graph)
highest_closeness_node = max(graph.nodes, key=closeness.get)
c=closeness[highest_closeness_node]
print("Highest Closeness Node: {} {}Closeness: {}".format(highest_closeness_node,'\n',c,'\n'))

eigenvector = nx.centrality.eigenvector_centrality(graph, max_iter = 800)
highest_eigenvector_node = max(graph.nodes, key=eigenvector.get)
e=eigenvector[highest_eigenvector_node]
print("Highest Eigan Vector Node: {} {}Eigan Vector: {}".format(highest_eigenvector_node, '\n', e))

#7
def findCommunities(graph):
    
    im = Infomap("--two-level --directed")
    print("Building Infomap network from a NetworkX graph...")
    
    for e in graph.edges():
        im.addLink(*e)
    print("Find communities with Infomap...")
    
    im.run();
    print("Found {0} communities with codelength: {1}".format(im.num_top_modules,
    im.codelength))
    communities = {}
    
    for node in im.tree:
        communities[node.node_id] = node.module_id
    nx.set_node_attributes(graph, communities,'community')
    
    return im.num_top_modules, communities

v=findCommunities(graph)
print("Found", v[0], "communities")

#8
c = Counter(v[1].values())

largest_community = sorted(c, key = c.get, reverse = True)[:1]
smallest_community = sorted(c, key = c.get, reverse = False)[:1]

print("Largest community:", largest_community)
print("Smallest community:", smallest_community)

#9
largest_communities = sorted(c, key=c.get, reverse=True)[:3]
print("The highest 3 communities are", largest_communities)

for i in largest_communities:
    selected_data = dict((n, d['community']) for n, d in graph.nodes().items() if d['community'] == i)
    sg = graph.subgraph(list(selected_data.keys()))
    
    fig, ax = plt.subplots()
    
    pos = nx.spring_layout(sg, seed=5656)
    nx.draw(sg, pos=pos, ax=ax)
    
    plt.title(f"Community {i}")
    plt.show()


# degree centrality
empty_list = []

for i in largest_communities:
    selected_data = dict((n, d['community']) for n, d in graph.nodes().items() if d['community'] == i)
    sg = graph.subgraph(list(selected_data.keys()))
    
    degree_centrality = nx.degree_centrality(sg).items()
    nodes_to_delete = sorted(degree_centrality, key=lambda pair: -pair[1])[:3]
    nodes_to_delete = [i[0] for i in nodes_to_delete]
    
    for node in nodes_to_delete:
        empty_list.append(node)
                        
    color_ = ["yellow" if node in nodes_to_delete else "blue" for node in sg]
    pos = nx.spring_layout(sg, seed=5656)
    
    fig, ax = plt.subplots()
    
    nx.draw(sg, pos=pos, node_color=color_, ax=ax)
    plt.title(f"Community {i}")
    
    plt.show()
    plt.close()


#closeness centrality
empty_list = []

for i in largest_communities:
    selected_data = dict((n, d['community']) for n, d in graph.nodes().items() if d['community'] == i)
    sg = graph.subgraph(list(selected_data.keys()))

    closeness_centrality = nx.closeness_centrality(sg).items()
    nodes_to_delete = sorted(closeness_centrality, key=lambda pair: -pair[1])[:3]
    nodes_to_delete = [i[0] for i in nodes_to_delete]

    for node in nodes_to_delete:
        empty_list.append(node)

    color_ = ["yellow" if node in nodes_to_delete else "blue" for node in sg]
    pos = nx.spring_layout(sg, seed=5656)

    fig, ax = plt.subplots()
    nx.draw(sg, pos=pos, node_color=color_, ax=ax)
    plt.title(f"Community {i}")

    plt.show()
    
    plt.close()


#betweenness centrality
empty_list = []

for i in largest_communities:
    selected_data = dict((n, d['community']) for n, d in graph.nodes().items() if d['community'] == i)
    sg = graph.subgraph(list(selected_data.keys()))

    betweenness_centrality = nx.betweenness_centrality(sg).items()
    nodes_to_delete = sorted(betweenness_centrality, key=lambda pair: -pair[1])[:3]
    nodes_to_delete = [i[0] for i in nodes_to_delete]

    for node in nodes_to_delete:
        empty_list.append(node)

    color_ = ["yellow" if node in nodes_to_delete else "blue" for node in sg]
    pos = nx.spring_layout(sg, seed=5656)

    fig, ax = plt.subplots()
    nx.draw(sg, pos=pos, node_color=color_, ax=ax)
    plt.title(f"Community {i}")

    plt.show()
    
    plt.close()


#eigenvector_centrality
empty_list = []

for community in largest_communities:
    selected_data = dict((n, d['community']) for n, d in graph.nodes().items() if d['community'] == community)
    sg = graph.subgraph(list(selected_data.keys()))

    eigenvectors_centrality = nx.centrality.eigenvector_centrality(sg, max_iter=1300)
    eigenvectors_ = nx.centrality.eigenvector_centrality(sg.reverse(), max_iter=1300)

    for node in eigenvectors_centrality.keys():
        eigenvectors_centrality[node] = (eigenvectors_centrality[node] + eigenvectors_[node]) / 2

    nodes_to_delete = sorted(eigenvectors_centrality, key=lambda pair: -eigenvectors_centrality[pair])[:3]

    for i in nodes_to_delete:
        empty_list.append(i)

    color_ = ["yellow" if node in nodes_to_delete else "blue" for node in sg]
    pos = nx.spring_layout(sg, seed=5656)

    fig, ax = plt.subplots()
    nx.draw(sg, pos=pos, node_color=color_, ax=ax)

    plt.show()
    plt.close()


#10
influencers_list = set(empty_list)

pos = nx.spring_layout(graph, seed=5656)
color_ = ['yellow' if node in influencers_list else "blue" for node in graph]

fig, ax = plt.subplots()
nx.draw(graph, pos=pos, node_color=color_, ax=ax)

plt.show()