# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:13:11 2018

@author: dmbes
"""

def graph_centrality(graph, kind = 'degree'):
    import networkx as nx
    import numpy as np
    if kind == 'degree':
        d = list(dict(graph.degree).values())
    if kind == 'betweenness':
        d = list(dict(nx.betweenness_centrality(graph, k = 500, normalized = False)).values())
    n = len(d)
    d_bar = np.max(d)
    d_all = []
    for i in d:
        d_all.append(d_bar - i)
    return(sum(d_all)/(n-2)) 
#%%
    
def check_directory(user_id, directory, kind = 'json'):
    files = os.listdir(directory)
    final = []
    for f in files:
        if str(user_id) in f:
            if kind in f:
                final.append(f)
    return(final)
            
#%%
def get_simmelian_ties(graph):
    '''
    This function calculates the total number of Simmelian Ties as presented
    by Krackhardt (1998).  This implementation uses the enhancements presented
    by Dekker (2006).
    '''
    import networkx as nx
    import numpy as np
    if not graph.is_directed():
        sys.exit('Graph is not directed')
    union = graph.to_undirected(reciprocal = True)
    Y = nx.to_numpy_matrix(union)
    Y2 = np.matmul(Y, Y)
    S = np.multiply(Y,Y2)
    
    return((S > 0).sum())
#%%
    
    
def get_user_data(api, user_id, directory, random_seed = None):
    import progressbar
    import random
    tweets = []
    try:
        tweets.extend(get_timeline(api, user_id, directory))
        frd = get_followers(api, user_id, directory)
    except:
        print('Could not scrape:',user_id)
        return([])
    random.seed(a=random_seed)
    bar = progressbar.ProgressBar()
    if len(frd ) > 0:
        for f in bar(random.sample(frd, min(len(frd),250))):
            try:
                tweets.extend(get_timeline(api, f, directory))
            except:
                continue
        
    return(dedupe_twitter(tweets))
#%%

def dedupe_twitter(list_of_tweets):
    seen = {}
    final = []
    for tweet in list_of_tweets:
        try:
            id = tweet["id"]
            if id not in seen:
                seen[id] = True
                final.append(tweet)
        except:
            continue
        
    return(final)
#%%
def get_timeline(api, user_id, directory):
    import gzip, json, io
    files = check_directory(user_id, directory, kind = '.json')
    timeline = []
    if len(files) > 0:
        file = files[0]
        if '.gz' in file:
            infile = io.TextIOWrapper(gzip.open(directory + '/' + file, 'r'))
        else:
            infile = open(directory + '/' + file, 'r')
        for line in infile:
            timeline.append(json.loads(line))
    else:
        new_tweets = api.user_timeline(id = user_id,count=200)
        outfile = gzip.open(directory + '/' + str(user_id) + '.json.gz', 'wt')
        for tweet in new_tweets:
            timeline.append(tweet)
            out = json.dumps(tweet._json)
            outfile.write(out + '\n')
            timeline.append(json.loads(tweet._json))
        outfile.close()

        
    if len(timeline) > 200:
        timeline = timeline[:200]
    return(timeline)
        
        
    
#%%
def get_followers(api, user_id, directory):
    import pandas as pd
    files = check_directory(user_id, directory, kind = '_followers.csv')
    if len(files) > 0:
        file = files[0]
        followers = pd.read_csv(directory + '/' + file, dtype = str)
        followers = followers[0].tolist()
    else:
        followers = api.followers_ids(id = user_id)
    return(followers)
    
    
#%%

def parse_all_metrics(api, edge_df, user_id, directory=None):
    import pandas as pd
    import twitter_col
    import json, io, gzip, os
    import dateutil
    from datetime import datetime, timezone
    import progressbar
    import networkx as nx
    from collections import Counter
    import community
    import numpy as np
    
#    user_id = '1919751'
    G=nx.from_pandas_edgelist(edge_df, 'from', 'to', edge_attr=['type'], create_using=nx.DiGraph())
#    G=nx.gnp_random_graph(100, 0.4, seed=None, directed=True)
    G2 = G.to_undirected()
    
    largest_component = max(nx.connected_component_subgraphs(G2), key=len)
    
    print("Nodes in largest compo:",len(largest_component.nodes))
    
    data = { 
            "user_id": [],
            "scrape_date":[],
            "num_nodes" : [],
            "num_links": [],
            "density": [],
            "isolates": [],
            "dyad_isolates": [],
            "triad_isolates": [],
            "compo_over_4": [],
#            "average_shortest_path_length": [],
            "clustering_coefficient": [],
            "transitivity": [],
#            "network_diameter": [],
            "reciprocity": [],
            "graph_degree_centrality": [],
            "graph_betweenness_centrality":[],
            "mean_eigen_centrality": [],
            "simmelian_ties": [],
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
    
    data['user_id'].append(user_id)
    data['scrape_date'].append(time.strftime('%Y%m%d-%H%M%S'))
    data['num_nodes'].append(nx.number_of_nodes(G))
    data['num_links'].append(nx.number_of_edges(G))
    data['density'].append(nx.density(G))
    
    compo_sizes = [len(c) for c in sorted(nx.connected_components(G2), key=len, reverse=True)]
    compo_freq = Counter(compo_sizes)
    
#    print('isolates')
    data['isolates'].append(compo_freq[1])
#    print('triad_islolates')
    data['triad_isolates'].append(compo_freq[3])
    data['dyad_isolates'].append(compo_freq[2])
    data['compo_over_4'].append(len([x for x in compo_sizes if x > 3]))
#    print('shortest path')
#    data['average_shortest_path_length'].append(nx.average_shortest_path_length(largest_component))
#    print('clustering_coefficient')
    data['clustering_coefficient'].append(nx.average_clustering(G2))
#    print('transitivity')
    data['transitivity'].append(nx.transitivity(G))
#    print('diameter')
#    data['network_diameter'].append(nx.diameter(largest_component))
#    print('reciprocity')
    data['reciprocity'].append(nx.reciprocity(G))
    
    ef = nx.effective_size(G, nodes = [user_id])
#    print('effective_size')
    data['ego_effective_size'].append(ef[user_id])
    
#    print('degree')
    data['graph_degree_centrality'].append(graph_centrality(G, kind = 'degree'))
#    print('betweenness')
    data['graph_betweenness_centrality'].append(graph_centrality(largest_component, kind = 'betweenness'))
#    print('eigen_centrality')
    eig = list(nx.eigenvector_centrality_numpy(G).values())
    data['mean_eigen_centrality'].append(np.mean(eig))
    
#    print('simmelian')
    data['simmelian_ties'].append(get_simmelian_ties(G))
#    print('census')
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
#    print('louvaine')
    data['num_louvaine_groups'].append(len(set(partition.values())))
    data['size_largest_louvaine_group'].append(p_df[0].value_counts().max())
    
    df = pd.DataFrame(data)
    return(df)
    
    
#%%
def get_metrics_listOfIDs(list_of_user_ids, api, directory, 
                          file_prefix = 'twitter_network_metrics_',
                          RS = None):
    import twitter_col
    import pandas as pd
    import progressbar
    
    ## run one iteration and create CSV
    data = get_user_data(api, list_of_user_ids[0], directory, random_seed = RS)
    edge = twitter_col.get_edgelist_from_list(data, to_csv = False, index = False)
    if len(edge.index) > 3:
        metric_df = parse_all_metrics(api, edge, list_of_user_ids[0], directory)
        metric_df.to_csv(file_prefix + 'network_features.csv')
    
    ## Loop through rest of IDs and append to CSV
    bar = progressbar.ProgressBar()
    with open(file_prefix + 'network_features.csv', 'a') as myFile:
        for user in bar(list_of_user_ids[1:]):
            data = get_user_data(api, user, directory, random_seed = RS)
            edge = twitter_col.get_edgelist_from_list(data, to_csv = False)
            if len(edge.index) > 3:
                metric_df = parse_all_metrics(api, edge, user, directory)
                metric_df.to_csv(myFile, header=False, index = False)
    
#%%
#import networkx as nx
#G=nx.gnp_random_graph(100, 0.4, seed=None, directed=True)
#
#d = nx.triadic_census(H)
#
#H = nx.gnp_random_graph(4, 0.4, seed=None, directed=True)
#
#d = dict(G.degree).values()
#
#partition = community.best_partition(G.to_undirected())
#
#ef = nx.effective_size(G, nodes = [user_id])
#df = parse_all_metrics(G, user_id = 50, directory = None)
#
##%%
#import tweepy
#import time
#import sys
#import json
#import os
#
## Replace the API_KEY and API_SECRET with your application's key and secret.
#auth = tweepy.AppAuthHandler("***REMOVED***", "***REMOVED***")
#
#api = tweepy.API(auth, wait_on_rate_limit=True,
#				   wait_on_rate_limit_notify=True)
#
#if (not api):
#    print ("Can't Authenticate")
#    sys.exit(-1)
##%%
#    
#test = get_user_data(api, '1919751', 'test')
#import twitter_col
#edge = twitter_col.get_edgelist_from_list(test, to_csv = False)
#metric_df = parse_all_metrics(api, edge, directory=None)
##%%
#import pandas as pd
#df = pd.read_csv('sn_id_lookup.csv')
#test = df['id_str'].tolist()
#test = test[:7]
#
#get_metrics_listOfIDs(test, api, 'test', 
#                          file_prefix = 'twitter_network_metrics_',
#                          RS = 777)
#
##%%
#df2 = pd.read_csv('twitter_network_metrics_network_features.csv', dtype = str)
