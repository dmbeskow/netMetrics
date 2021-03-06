# -*- coding: utf-8 -*-
"""
Created on Thu Mar 29 10:13:11 2018

@author: dmbes
"""

import twitter_col
import pandas as pd
import time
import progressbar
import random
import tweepy
import networkx as nx
import numpy as np
import community
#%%

def graph_centrality(graph, kind = 'degree'):
    '''

    Parameters
    ----------
    graph : neworkx graph
    kind : 'degree' or 'betweenness'
        DThe default is 'degree'.

    Returns
    -------
    Graph Centrality

    '''

    if kind == 'degree':
        d = list(dict(graph.degree).values())
    if kind == 'betweenness':
        d = list(dict(nx.betweenness_centrality(graph, k = min(graph.number_of_nodes(),500), normalized = False)).values())
    n = len(d)
    d_bar = np.max(d)
    d_all = []
    for i in d:
        d_all.append(d_bar - i)
    return(sum(d_all)/(n-2))
#%%

def check_directory(user_id, directory, kind = 'json'):
    '''
    Check for files in directory
    '''
    import os
    files = os.listdir(directory)
    final = []
    for f in files:
        if str(user_id) in f:
            if kind in f:
                final.append(f)
    return(final)

#%%
def get_simmelian_ties(graph, sparse = False):
    '''
    This function calculates the total number of Simmelian Ties as presented
    by Krackhardt (1998).  This implementation uses the enhancements presented
    by Dekker (2006).
    '''
    import networkx as nx
    import numpy as np
    import scipy as sp
    import sys

    if not graph.is_directed():
        sys.exit('Graph is not directed')
    union = graph.to_undirected(reciprocal = True)

    if sparse:
        Y = nx.to_scipy_sparse_matrix(union)
        Y2 = sp.sparse.csr_matrix.dot(Y, Y)
        S = sp.sparse.csr_matrix.multiply(Y,Y2)
    else:
        Y = nx.to_numpy_matrix(union)
        Y2 = np.matmul(Y, Y)
        S = np.multiply(Y,Y2)

    return((S > 0).sum())
#%%


def get_user_data(api, user_id, directory, random_seed = None):
    '''
    Gets all required tier3 data for a given user

    Parameters
    ----------
    api : Tweepy API hook
    user_id : string of user ID
    directory : Location to save/look for data
    random_seed : seed for keeping same sample for followers
        DESCRIPTION. The default is None.

    Returns
    -------
    None.

    '''

    tweets = []
    try:
        tweets.extend(get_timeline(api, user_id, directory))
        frd = get_followers(api, user_id, directory)
    except tweepy.TweepError as e:
        print('Could not scrape:',user_id)
        print(e)
        return([])
    random.seed(a=random_seed)
    if len(frd ) > 0:
        for f in random.sample(frd, min(len(frd),250)):
            try:
                tweets.extend(get_timeline(api, f, directory))
            except:
                continue

    return(dedupe_twitter(tweets))
#%%


def timeline_snowball(api, user_id, directory, random_seed = None):
    '''
    Gets timeline for user and user followers (up to 250...more than 250, samples 250 from last 5000)

    '''
    import progressbar
    import random
    import tweepy
    try:
        get_timeline(api, user_id, directory)
        frd = get_followers(api, user_id, directory)
    except tweepy.TweepError as e:
        print('Could not scrape:',user_id)
        print(e)
        return([])
    random.seed(a=random_seed)
    bar = progressbar.ProgressBar()
    if len(frd ) > 0:
        for f in bar(random.sample(frd, min(len(frd),250))):
            try:
                get_timeline(api, f, directory)
            except:
                continue




#%%
def dedupe_users(list_of_users):
    '''
    Dedupes list of users

    Parameters
    ----------
    list_of_users : List of Tweets

    Returns
    -------
    Deduped list of users

    '''
    seen = {}
    final = []
    for tweet in list_of_users:
        try:
            id = tweet['user']['id_str']
            if id not in seen:
                seen[id] = True
                final.append(tweet)
        except:
            continue

    return(final)
#%%
def dedupe_twitter(list_of_tweets):
    '''
    Dedupe list of Tweets

    Parameters
    ----------
    list_of_tweets : List of tweet dictionaries

    Returns
    -------
    List of Deduped Tweets

    '''
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
def get_timeline(api, user_id, directory, pages = 1):
    '''
    Get timeline from user.  Use pages parameter to get up to last 3200 tweets
    in increments of 200 (1 page gives 200, 2 pages gives 400, etc.)
    '''
    import gzip, json, io
    import tweepy
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
        try:
            new_tweets = []
            for page in tweepy.Cursor(api.user_timeline, id=user_id, count = 200,tweet_mode="extended").pages(pages):
                new_tweets.extend(page)
            with gzip.open(directory + '/' + str(user_id) + '.json.gz', 'wt') as outfile:
                for tweet in new_tweets:
                    timeline.append(tweet)
                    out = json.dumps(tweet._json)
                    outfile.write(out + '\n')
                    timeline.append(tweet._json)
        except tweepy.TweepError as e:
            print(e.reason)

    return(timeline)



#%%
def get_followers(api, user_id, directory):
    '''
    Get Followers (Will check to make sure we haven't already)

    '''
    import pandas as pd
    files = check_directory(user_id, directory, kind = '_followers.csv')
    if len(files) > 0:
        file = files[0]
        followers = pd.read_csv(directory + '/' + file, dtype = str, header = None)
        followers = followers[0].tolist()
    else:
        followers = api.followers_ids(id = user_id)
        if len(followers) > 0:
            df = pd.DataFrame({'my_ids':followers}, dtype = str)
            df.to_csv(directory + '/' + user_id + '_followers.csv',header = None, index = False)
    return(followers)
#%%
def get_friends(api, user_id, directory):
    '''
      Get Followers (Will check to make sure we haven't already)

    '''
    import pandas as pd
    files = check_directory(user_id, directory, kind = '_friends.csv')
    if len(files) > 0:
        file = files[0]
        friends = pd.read_csv(directory + '/' + file, dtype = str, header = None)
        friends = friends[0].tolist()
    else:
        friends = api.friends_ids(id = user_id)
        df = pd.DataFrame({'my_ids':friends}, dtype = str)
        df.to_csv(directory + '/' + user_id + '_friends.csv',header = None, index = False)
    return(friends)

#%%%

def parse_all_metrics(api, edge_df, user_id, directory=None, long = False):
    '''
    Will get all Tier 3 metrics for a user_id

    Parameters
    ----------
    api : Tweepy API hook
    edge_df : Edgelist of Pandas DataFrame
    user_id : User ID string
    directory : Directory to look for data
        The default is None.
    long : Whether to get metrics that take a long time. The default is False.

    Returns
    -------
    Feature Data Frame

    '''
    import pandas as pd
    import twitter_col
    import json, io, gzip, os
    import time
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

    if long:
        data.pop("graph_betweenness_centrality")
        data.pop("ego_effective_size")
        data.pop("simmelian_ties")

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
#    print('effective size')
    if not long:
        if user_id in list(G.nodes):
            ef = nx.effective_size(G, nodes = [user_id])
            data['ego_effective_size'].append(ef[user_id])
        else:
            data['ego_effective_size'].append(0)

#    print('degree')
    data['graph_degree_centrality'].append(graph_centrality(G, kind = 'degree'))
#    print('betweenness')
    if not long:
        data['graph_betweenness_centrality'].append(graph_centrality(largest_component, kind = 'betweenness'))
#    print('eigen_centrality')
    try:
        eig = list(nx.eigenvector_centrality_numpy(G).values())
        data['mean_eigen_centrality'].append(np.mean(eig))
    except:
        data['mean_eigen_centrality'].append(0)

#    print('simmelian')
#    if long:
    data['simmelian_ties'].append(get_simmelian_ties(G, sparse = True))
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
def get_metrics_listOfIDs(list_of_user_ids, api, directory, bot_model,
                          file_prefix = 'twitter_network_metrics_',
                          RS = None):
    '''
    Get metrics for list of IDS
    '''


    myTime = time.strftime('%Y%m%d-%H%M%S')

    ## run one iteration and create CSV
    print('Getting Data for ',list_of_user_ids[0])
    data = get_user_data(api, list_of_user_ids[0], directory, random_seed = RS)
    print('Building features...')
    edge = twitter_col.get_edgelist_from_list(data, to_csv = False)
    if len(edge.index) > 3:
        metric_df = parse_all_metrics(api, edge, list_of_user_ids[0], directory)
        content_df = get_network_user_data(data, list_of_user_ids[0], bot_model)
        final_df = pd.merge(metric_df, content_df, how = 'inner', on = 'user_id')
        final_df.to_csv(file_prefix + 'network_features' + myTime + '.csv', index = False)

    ## Loop through rest of IDs and append to CSV

    with open(file_prefix + 'network_features' + myTime + '.csv', 'a') as myFile:
        bar = progressbar.ProgressBar()
        for user in bar(list_of_user_ids[1:]):
            data = get_user_data(api, user, directory, random_seed = RS)
            edge = twitter_col.get_edgelist_from_list(data, to_csv = False)
            if len(edge.index) > 3:
#                try:
                metric_df = parse_all_metrics(api, edge, user, directory)
                content_df = get_network_user_data(data, user, bot_model)
                final_df = pd.merge(metric_df, content_df, how = 'inner', on = 'user_id')
                final_df.to_csv(myFile, index = False, header = False)
#                except:
#                    continue

#%%
def get_network_user_data(data, user_id, bot_model ):
    ''' Get Tier2 metrics for a user
    

    Parameters
    ----------
    data : list of twitter data
    user_id : user_id (string)
    bot_model : Model to use for Bot-Hunter Tier 1 detection

    Returns
    -------
    Pandas data frame of features

    '''
    import pandas as pd
    from datetime import datetime, timezone
    import dateutil
    import twitter_col
    import numpy as np
    from sklearn.feature_extraction.text import CountVectorizer
    from scipy.spatial.distance import squareform, pdist
    from scipy import stats
    from botHunter import classification
    import json

    final = {'user_id' : [],
            'network_fraction_default_image' : [],  #
             'network_median_followers' : [], #
             'network_median_friends' : [], #
             'network_number_of_languages': [], #
             'network_median_statuses': [], #
             'network_mean_age' : [], #
             'network_fraction_with_description' : [], #
             'network_mean_num_emoji' : [],#
             'network_mean_num_emoji_original' : [],#
             'network_mean_mentions' : [],#
             'network_mean_hash' : [],#
             'network_fraction_retweets' : [],#
             'network_mean_tweets_per_min' : [],#
             'network_mean_tweets_per_hour' : [],#
             'network_mean_tweets_per_day' : [],#
             'network_mean_jaccard_similarity' : [],#
             'network_mean_cosine_similarity': [],
             'network_sleep_at_night' : [],
             'network_unpopAcct_popTeet': [],
             'network_bot_percentage': []
             }

    final['user_id'].append(user_id)
    df = twitter_col.parse_twitter_list(data)
    df['date2'] = twitter_col.convert_dates(df['status_created_at'].tolist())
    df['hour'] = df.date2.dt.strftime('%H').astype(int)
    df.index = pd.DatetimeIndex(df.date2)
    df['hash'] = get_num_hash(data)
    df['mention'] = get_num_mention(data)
    df['emoji'] = df['status_text'].apply(get_num_emoji)


    # Temporal data

    final["network_mean_tweets_per_min"].append(df['date2'].resample('T').count().mean())
    final["network_mean_tweets_per_hour"].append(df['date2'].resample('H').count().mean())
    final["network_mean_tweets_per_day"].append(df['date2'].resample('D').count().mean())

    # sleep analysis
    # see https://machinelearningmastery.com/time-series-data-stationary-python/
    # and http://www.statsmodels.org/dev/generated/statsmodels.tsa.stattools.adfuller.html
    # I ended up using KS-test
    # To plot one of their historgrams, use:
    # timeOfDay['428454668'].hist()

    timeOfDay = df.groupby('id_str')['hour'].apply(ks_test_uniformity)

    final["network_sleep_at_night"].append(np.sum(timeOfDay > 0.5)/len(timeOfDay))

    # Text data
    final['network_mean_num_emoji'].append(df['emoji'].mean())
    final['network_mean_num_emoji_original'].append(df[df['status_isretweet'] == False]['emoji'].mean())
    final['network_mean_mentions'].append(df['mention'].mean())
    final['network_mean_hash'].append(df['hash'].mean())
    final['network_fraction_retweets'].append(sum(df['status_isretweet'] == True)/len(df.index))

    # Jaccard and Cosine Similarity
    if user_id in df['id_str'].tolist():
        text = df.groupby(['id_str'])['status_text'].apply(','.join).reset_index()
        text['status_text'] = text['status_text'].str.replace('http\S+|www.\S+', '', case=False)
        docs = text['status_text'].tolist()
        vec = CountVectorizer()
        X = vec.fit_transform(docs)
        text3 = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
        text3.index = text['id_str']
        dist = pdist(text3, metric="jaccard")
        dist2 = squareform(dist)
        index = text3.index.get_loc(user_id)
        final['network_mean_jaccard_similarity'].append(np.mean(dist2[index,]))
        dist = pdist(text3, metric="cosine")
        dist2 = squareform(dist)
        dist2 = np.nan_to_num(dist2)
        index = text3.index.get_loc(user_id)
        final['network_mean_cosine_similarity'].append(np.mean(dist2[index,]))
    else:
        final['network_mean_jaccard_similarity'].append(1)
        final['network_mean_cosine_similarity'].append(1)


    # User Data
    df2 = df.drop_duplicates(subset = ['id_str'], keep = 'first')
    df2 = df2.reset_index(drop = True)
    df2['age'] = df2['created_at'].apply(get_age)

    final['network_fraction_default_image'].append(sum(df2['has_default_profile'] == True)/len(df2.index))
    final['network_median_followers'].append(df2['followers_count'].median())
    final['network_median_friends'].append(df2['friends_count'].median())
    final['network_median_statuses'].append(df2['friends_count'].median())

    df['max'] = df[['followers_count','friends_count']].astype(int).max(axis = 1)
    df['pop_unpop'] = df['status_retweet_count'].astype(int) > 2 * df['max'].astype(float)
    final['network_unpopAcct_popTeet'].append(df['pop_unpop'].sum()/len(df.index))

    final['network_fraction_with_description'].append(sum(df2['has_default_profile'] == True)/len(df2.index))

    final['network_mean_age'].append(df2['age'].mean())
    final['network_number_of_languages'].append(df2['lang'].value_counts().count())

        # Get bots
    with open('temp.json', 'w') as outfile:
        for tweet in dedupe_users(data):
            outfile.write(json.dumps(tweet) + '\n')
    bots = classification.bot_classification('temp.json', model = bot_model)
    final['network_bot_percentage'].append(sum(bots['prediction'])/len(bots.index))

    df = pd.DataFrame(final)
    return(df)


#%%
def check_tweet(Tweet):
    '''
    Flips users to tweet format and checks to make sure there's a status object.
    If no status object, added an empty status object.
    
    '''
    import twitter_col
    if 'status' not in Tweet.keys():
        if 'friends_count' in Tweet.keys():
            Tweet['status'] = twitter_col.get_empty_status()
    if 'status' in Tweet.keys():
        temp = Tweet['status']
        getRid = Tweet.pop('status', 'Entry not found')
        temp['user'] = Tweet
        Tweet = temp
        return(Tweet)
#%%
def get_num_hash(tweets):
    """
    Returns number of hashtags in a tweet.  If no hashtags,
    it returns an empty list.

    """
    final = []
    for tweet in tweets:
        ht = []
        for h in tweet['entities']['hashtags']:
                ht.append(h['text'])
        final.append(len(ht))
    return(final)


#%%
def get_num_mention(tweets):
    """
    Returns number of mentions in a tweet.  If no hashtags,
    it returns an empty list.
    """
    final = []
    for tweet in tweets:
        men = []
        if len(tweet['entities']['user_mentions']) > 0:
            for m in tweet['entities']['user_mentions']:
                men.append(m['id_str'])
        final.append(len(men))
    return(final)
#%%
def get_num_emoji(string):
    """
   Returns number of emojis

    """
    import emoji
    E = [c for c in string if c in emoji.UNICODE_EMOJI]
    return(len(E))
#%%
def get_age(date):
    '''
    Return age of past date.

    '''
    from datetime import datetime, timezone
    import dateutil
    td = datetime.now(timezone.utc) - dateutil.parser.parse(date)
    return(td.days)
#%%
def my_adfuller(x):
    '''
    Returns adfuller metric

    '''
    from statsmodels.tsa.stattools import adfuller
#    vector = vector.as_matrix()
    try:
        result = adfuller(x)
        return(result[1])
    except:
        return(0.4)
#%%
#timeOfDay = df['hour'].value_counts().as_matrix()
def ks_test_uniformity(timeOfDay):
    '''
    Conducts KS-test for uniformity

    '''
    from scipy import stats
    try:
        y1 = timeOfDay - min(timeOfDay)
        z1 = timeOfDay/max(y1)
        statistic,pvalue = stats.kstest(z1, 'uniform')
        return(pvalue)
    except:
        return(0.4)


#%%
#import netMetrics
#d = netMetrics.get_user_data(api, '59220577','netMetric_timelines2',random_seed = 775)
##%%
### Jaccard similarity
#df2 = df.groupby(['id_str'])['status_text'].apply(','.join).reset_index()
#df2['status_text'] = df2['status_text'].str.replace('http\S+|www.\S+', '', case=False)
#import pandas as pd
#from sklearn.feature_extraction.text import CountVectorizer
#from scipy.spatial.distance import squareform, pdist
##docs = ['why hello there', 'omg hello pony', 'she went there? omg']
#docs = df2['status_text'].tolist()
#vec = CountVectorizer()
#X = vec.fit_transform(docs)
#df3 = pd.DataFrame(X.toarray(), columns=vec.get_feature_names())
#df3.index = df2['id_str']
#dist = pdist(df3, metric="jaccard")
#dist2 = squareform(dist)
#
#df3.index.get_loc('59220577')
##%%
#import json, io, gzip
#tweets = []
#with io.TextIOWrapper(gzip.open('nato_bot_all_20180222.json.gz', 'r')) as infile:
#    for line in infile:
#        if line != '\n':
#            tweets.append(json.loads(line))


#%%
#from scipy import stats
#import numpy as np
#
#x = np.random.normal(size = 50, loc = 30, scale = 3)
##x = np.random.uniform(size = 24, low = 25, high = 30)
#y = x - min(x)
#z = y/max(y)
#stats.kstest(z, 'uniform')

#%%

#    from statsmodels.tsa.stattools import adfuller
##    X = df['date2'].resample('H').count().as_matrix()
#    X = df.groupby('id_str')['date2'].resample('D').count()
#    y = X.groupby('id_str').apply( my_adfuller)
#    y['stationary'] = y
#    result = adfuller(X)
#    p_value = result[1]
#
#    final_result = []
#    people = list(set(df['id_str'].tolist()))
#    for person in people:
#        temp = df[df['id_str'] == person]
#        if len(temp.index) > 50:
#            X = temp.groupby('id_str')['date2'].resample('D').count()
#            result = adfuller(X)
#            p_value = result[1]
#            final_result.append(p_value)
#        else:
#            final_result.append(1)

#%%
#def parse_twitter_list(List):
#    """
#    This parses 'tweet' json to a pandas dataFrame. 'name' should be either
#    'id_str' or 'screen_name'.  This will choose which object is selected for
#    reply and retweet id.
#    """
#    import pandas as pd
#    import json, time, io, gzip
#    import progressbar
#    import twitter_col
#
#    data = { "id_str" : [],
#        "screen_name" : [],
#        "location" : [],
#        "description" : [],
#        "protected" : [],
#        "verified" : [],
#        "followers_count" : [],
#        "friends_count" : [],
#        "listed_count" : [],
#        "favourites_count" : [],
#        "statuses_count" : [],
#        "created_at" : [],
#        "status_created_at" : [],
#        "description" : [],
#        "
#       "geo_enabled" : [],
#        "lang" : [],
#        "has_default_profile" : []
#          }
#
#    bar = progressbar.ProgressBar()
#    for item in bar(List):
#        if item != '\n':
#            t = check_tweet(item)
#
#            if 'user' in t.keys():
#                data['id_str'].append(t['user']['id_str'])
#                data['screen_name'].append(t['user']['screen_name'])
#                data['location'].append(t['user']['location'])
#                data['description'].append(t['user']['description'])
#                data['protected'].append(t['user']['protected'])
#                data['verified'].append(t['user']['verified'])
#                data['followers_count'].append(t['user']['followers_count'])
#                data['friends_count'].append(t['user']['friends_count'])
#                data['listed_count'].append(t['user']['listed_count'])
#                data['favourites_count'].append(t['user']['favourites_count'])
#                data['statuses_count'].append(t['user']['statuses_count'])
#                data['created_at'].append(t['user']['created_at'])
#                data['geo_enabled'].append(t['user']['geo_enabled'])
#                data['lang'].append(t['user']['lang'])
#                if t['user']['profile_image_url'] == "http://abs.twimg.com/sticky/default_profile_images/default_profile_normal.png":
#                    data['has_default_profile'].append(True)
#                else:
#                    data['has_default_profile'].append(False)
#
#    df = pd.DataFrame(data, dtype = str)
#
#    return(df)



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

##%%
#import netMetrics
#test = get_user_data(api, '113142532', 'cav_timelines')
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
