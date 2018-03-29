# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:13:11 2018

@author: dmbes
"""

def graph_centrality(graph, kind = 'degree'):
    import numpy as np
    if kind == 'degree':
        d = list(dict(G.degree).values())
    if kind == 'betweenness':
        d = list(dict(nx.betweenness_centrality(G, normalized = False)).values())
    n = len(d)
    d_bar = np.max(d)
    d_all = []
    for i in d:
        d_all.append(d_bar - i)
    return(sum(d_all)/(n-2))
#%%

def parse_all_metrics(G, user_id, directory):
    import pandas as pd
    import twitter_col
    import json, io, gzip
    import dateutil
    from datetime import datetime, timezone
    import progressbar
    import networkx as nx
    from collections import Counter
    import community
    import numpy as np
#    from networkx.algorithms import community
    
#    G=nx.Graph()
#    G=nx.from_pandas_dataframe(df, source, target, edge_attr=None, create_using=None)
    G=nx.gnp_random_graph(100, 0.4, seed=None, directed=True)
    G2 = G.to_undirected()
    
    data = { 
            "num_nodes" : [],
            "num_links": [],
            "density": [],
            "isolates": [],
            "dyad_isolates": [],
            "triad_isolates": [],
            "compo_over_4": [],
            "average_shortest_path_length": [],
            "clustering_coefficient": [],
            "transitivity": [],
            "network_diameter": [],
            "reciprocity": [],
            "graph_degree_centrality": [],
            "graph_betweenness_centrality":[],
            "mean_eigen_centrality": [],
            "mean_simmelian_ties": [],
            "triad_003": [],
            "triad_012": [],
            "triad_102": [],
            "triad_021D": [],
            "triad_021U": [],
            "triad_021C": [],
            "triad_111D": [],
            "triad_111U": [],
            "triad_030T": [],
            "triad_030C": [],
            "triad_201": [],
            "triad_120D": [],
            "triad_120U": [],
            "triad_120C": [],
            "triad_210": [],
            "triad_300": [],
            "num_louvaine_groups": [],
            "size_largest_louvaine_group": [],
            "ego_effective_size": []
    }
    

    data['num_nodes'].append(nx.number_of_nodes(G))
    data['num_links'].append(nx.number_of_edges(G))
    data['density'].append(nx.density(G))
    
    compo_sizes = [len(c) for c in sorted(nx.connected_components(G2), key=len, reverse=True)]
    compo_freq = Counter(compo_sizes)
    
    data['isolates'].append(compo_freq[1])
    data['triad_isolates'].append(compo_freq[3])
    data['dyad_isolates'].append(compo_freq[2])
    data['compo_over_4'].append(len([x for x in compo_sizes if x > 3]))
    data['average_shortest_path_length'].append(nx.average_shortest_path_length(G))
    data['clustering_coefficient'].append(nx.average_clustering(G2))
    data['transitivity'].append(nx.transitivity(G))
    data['network_diameter'].append(nx.diameter(G))
    data['reciprocity'].append(nx.reciprocity(G))
    
    ef = nx.effective_size(G, nodes = [user_id])
    data['ego_effective_size'].append(ef[user_id])
    
    
    data['graph_degree_centrality'].append(graph_centrality(G, kind = 'degree'))
    data['graph_betweenness_centrality'].append(graph_centrality(G, kind = 'betweenness'))
    
    eig = list(nx.eigenvector_centrality(G).values())
    data['mean_eigen_centrality'].append(np.mean(eig))
    
    
    data['mean_simmelian_ties'].append(None)
    
    census = nx.triadic_census(G)
    
    data['triad_003'].append(census['003'])
    data['triad_012'].append(census['012'])
    data['triad_102'].append(census['021C'])
    data['triad_021D'].append(census['021D'])
    data['triad_021U'].append(census['021U'])
    data['triad_021C'].append(census['030C'])
    data['triad_111D'].append(census['030T'])
    data['triad_111U'].append(census['102'])
    data['triad_030T'].append(census['111D'])
    data['triad_030C'].append(census['111U'])
    data['triad_201'].append(census['120C'])
    data['triad_120D'].append(census['120D'])
    data['triad_120U'].append(census['120U'])
    data['triad_120C'].append(census['201'])
    data['triad_210'].append(census['210'])
    data['triad_300'].append(census['300'])
    
    partition = community.best_partition(G2)
    p_df = pd.DataFrame.from_dict(partition, orient = 'index')
    
    data['num_louvaine_groups'].append(len(set(partition.values())))
    data['size_largest_louvaine_group'].append(p_df[0].value_counts().max())
    
    df = pd.DataFrame(data)
    return(df)
    
    
#%%
import networkx as nx
G=nx.gnp_random_graph(100, 0.4, seed=None, directed=True)

d = nx.triadic_census(H)

H = nx.gnp_random_graph(4, 0.4, seed=None, directed=True)

d = dict(G.degree).values()

partition = community.best_partition(G.to_undirected())

ef = nx.effective_size(G, nodes = [user_id])
df = parse_all_metrics(G, user_id = 50, directory = None)
