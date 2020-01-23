#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 23 08:29:37 2020

@author: david
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
def strip_all_entities(text):
    '''Replaces punctuation and strips out hashtags and mentions
    

    Parameters
    ----------
    text (string): text string

    Returns
    -------
    string: The input with punctuation replaced and hashtags and mentions removed

    '''
    import re, string
    entity_prefixes = ['@','#']
    for separator in  string.punctuation:
        if separator not in entity_prefixes :
            text = text.replace(separator,' ')
    words = []
    for word in text.split():
        word = word.strip()
        if word:
            if word[0] not in entity_prefixes:
                words.append(word)
    return ' '.join(words)




#%%
def network_triage(file, to_csv = False, languages = 'all'):
    ''' Finds top 10 Words and Hashtags by Louvain Group
    

    Parameters
    ----------
    file : path to file
    to_csv : Whether to write results to disk. The default is False.
    languages : Which language to use for stopwords.  The default is 'all'.

    Returns
    -------
    2 Pandas Dataframes: 1 Top Words and 1 for Top Hashtags

    '''
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import twitter_col
    import pandas as pd
    import progressbar
    import networkx as nx
    import community
    import string
    import nltk
    import json
    import urllib.request
    import re


    final_hash = {'hash_count': []}
    final_words = {'tweet_count': []}

    link = "https://raw.githubusercontent.com/dmbeskow/stop-words/master/stopwords-all.json"
    with urllib.request.urlopen(link) as url:
        stop_dict = json.loads(url.read().decode())

    stop_words = stopwords.words('english')

#    ukrain_stop_words = pd.read_csv('/Users/dbeskow/Dropbox/CMU/bot_classification/botApp/ukrainian-stopwords.txt',header = None)
#    ukrain_stop_words = pd.read_csv('/usr0/home/dbeskow/Dropbox/CMU/bot_classification/botApp/ukrainian-stopwords.txt',header = None)
#    stop_words.extend(ukrain_stop_words[0].tolist())
    if languages == 'all':
        for key in stop_dict:
            stop_words.extend(stop_dict[key])
    else:
        for item in languages:
            if item in stop_dict:
                stop_words.extend(stop_dict[item])
    stop_words.extend(string.punctuation)
    stop_words.extend(['rt', '@', '#', 'http', 'https', '!', '?', '(', ')','`', '’','``'])

    edge_df = twitter_col.get_edgelist_file(file, to_csv = False, kind = 'id_str')
    data = twitter_col.parse_only_text(file, to_csv = False)
    hashtags = twitter_col.extract_hashtags(file, to_csv = False)
    hashtags['user'] = hashtags['user'].astype(str)

    text_dict = {}
    for key, s in data.groupby('id_str')['status_text']:
        text_dict[key] = list(s)

    hash_dict = {}
    for key, s in hashtags.groupby('user')['hashtag']:
        hash_dict[key] = list(s)


    G=nx.from_pandas_edgelist(edge_df, 'from', 'to', edge_attr=['type','status_id', 'created_at'])

#    G.remove_node(None)
    partition = community.best_partition(G)
    p_df = pd.DataFrame.from_dict(partition, orient = 'index')
    table = p_df[0].value_counts()

    myMax = min(10,len(table.index))

    table = table.nlargest(myMax)

    groups = list(table.index)
    bar = progressbar.ProgressBar()
    for group in bar(groups):
        tweets = []
        Hash = []
        temp = p_df[p_df[0] == group]
        users = list(set(temp.index))
        bar2 = progressbar.ProgressBar()

        for u in bar2(users):
            if u in text_dict:
                tweets.extend(text_dict[u])
            if u in hash_dict:
                Hash.extend(hash_dict[u])

        tweets = list(map(lambda item: item.lower(), tweets))
        tweets = list(map(lambda item: strip_all_entities(item),tweets))
        tokenized_tweets = [word_tokenize(i) for i in tweets]
        words = [item for sublist in tokenized_tweets for item in sublist if item not in stop_words]
#        words = [item for sublist in tokenized_tweets for item in sublist]
        regex = re.compile('#(\w+)')
        words = [x for x in words if not regex.match(x)]
        regex = re.compile('@(\w+)')
        words = [x for x in words if not regex.match(x)]
        allWordDist = nltk.FreqDist(w.lower() for w in words)
        common_words = allWordDist.most_common(10)
        common_words = [x[0] for x in common_words]

        allWordDist = nltk.FreqDist(w.lower() for w in Hash)
        common_hash = allWordDist.most_common(10)
        common_hash = [x[0] for x in common_hash]

        final_words['tweet_count'].append(len(tweets))
        final_hash['hash_count'].append(len(Hash))

        final_words[group] = common_words
        final_hash[group] = common_hash

    t_count = final_words.pop('tweet_count')
    words_df = pd.DataFrame(final_words)
    words_df = words_df.transpose()
    words_df =  pd.merge(words_df,table.to_frame('node_count'), left_index = True, right_index = True)
    nc = words_df['node_count']
    words_df = words_df.drop('node_count', axis = 1)
    words_df.insert(0, 'node_count', nc)
    words_df.insert(0, 'tweet_count', t_count)
    words_df.insert(0, 'group', words_df.index)


    for key in final_hash:
        while len(final_hash[key]) < 10:
            final_hash[key].append(None)
    h_count = final_hash.pop('hash_count')
    hash_df = pd.DataFrame(final_hash)
    hash_df = hash_df.transpose()
    hash_df = pd.merge(hash_df,table.to_frame('node_count'), left_index = True, right_index = True)
    nc = hash_df['node_count']
    hash_df = hash_df.drop('node_count', axis = 1)
    hash_df.insert(0, 'node_count', nc)
    hash_df.insert(0, 'hash_count', h_count)
    hash_df.insert(0, 'group', hash_df.index)

    if to_csv:
        words_df.to_csv('wordTriage_' + file + '.csv',index = False)
        hash_df.to_csv('hashTriage_' + file + '.csv', index = False)
    else:
        return(words_df, hash_df)


#%%
def word_triage_by_language(file, to_csv = False, languages = 'all'):
    ''' Find top words and hashtags by language
    

    Parameters
    ----------
    file : path to file
    to_csv : Whether to write to disk.  The default is False.
    languages : Languages for stopwords.  The default is 'all'.

    Returns
    -------
    2 Pandas Dataframes: 1 Top Words and 1 for Top Hashtags

    '''
    from nltk.tokenize import word_tokenize
    from nltk.corpus import stopwords
    import twitter_col
    import pandas as pd
    import progressbar
    import string
    import nltk
    import json
    import urllib.request
    import re


    final_hash = {'hash_count': []}
    final_words = {'tweet_count': []}

    link = "https://raw.githubusercontent.com/dmbeskow/stop-words/master/stopwords-all.json"
    with urllib.request.urlopen(link) as url:
        stop_dict = json.loads(url.read().decode())

    stop_words = stopwords.words('english')

#    ukrain_stop_words = pd.read_csv('/Users/dbeskow/Dropbox/CMU/bot_classification/botApp/ukrainian-stopwords.txt',header = None)
#    ukrain_stop_words = pd.read_csv('/usr0/home/dbeskow/Dropbox/CMU/bot_classification/botApp/ukrainian-stopwords.txt',header = None)
#    stop_words.extend(ukrain_stop_words[0].tolist())
    if languages == 'all':
        for key in stop_dict:
            stop_words.extend(stop_dict[key])
    else:
        for item in languages:
            if item in stop_dict:
                stop_words.extend(stop_dict[item])
    stop_words.extend(string.punctuation)
    stop_words.extend(['rt', '@', '#', 'http', 'https', '!', '?', '(', ')','`', '’','``'])

    #edge_df = twitter_col.get_edgelist_file(file, to_csv = False)
    data = twitter_col.parse_only_text(file, to_csv = False)
    data = data[['id_str','status_text','lang']]
    hashtags = twitter_col.extract_hashtags(file, to_csv = False)
    hashtags['user'] = hashtags['user'].astype(str)

    text_dict = {}
    for key, s in data.groupby('id_str')['status_text']:
        text_dict[key] = list(s)

    hash_dict = {}
    for key, s in hashtags.groupby('user')['hashtag']:
        hash_dict[key] = list(s)


    #G=nx.from_pandas_edgelist(edge_df, 'from', 'to', edge_attr=['type','status_id', 'created_at'])
    #G.remove_node(None)
    #partition = community.best_partition(G)
    #p_df = pd.DataFrame.from_dict(partition, orient = 'index')
    table = data['lang'].value_counts()

    myMax = min(10,len(table.index))

    table = table.nlargest(myMax)

    groups = list(table.index)
    bar = progressbar.ProgressBar()
    for group in bar(groups):
        tweets = []
        Hash = []
        temp = data[data['lang'] == group]
        users = temp['id_str'].tolist()
        bar2 = progressbar.ProgressBar()

        for u in bar2(users):
            if u in text_dict:
                tweets.extend(text_dict[u])
            if u in hash_dict:
                Hash.extend(hash_dict[u])

        tweets = list(map(lambda item: item.lower(), tweets))
        tweets = list(map(lambda item: strip_all_entities(item),tweets))
        tokenized_tweets = [word_tokenize(i) for i in tweets]
        words = [item for sublist in tokenized_tweets for item in sublist if item not in stop_words]
#        words = [item for sublist in tokenized_tweets for item in sublist]
        regex = re.compile('#(\w+)')
        words = [x for x in words if not regex.match(x)]
        regex = re.compile('@(\w+)')
        words = [x for x in words if not regex.match(x)]
        allWordDist = nltk.FreqDist(w.lower() for w in words)
        common_words = allWordDist.most_common(10)
        common_words = [x[0] for x in common_words]

        allWordDist = nltk.FreqDist(w.lower() for w in Hash)
        common_hash = allWordDist.most_common(10)
        common_hash = [x[0] for x in common_hash]

        final_words['tweet_count'].append(len(tweets))
        final_hash['hash_count'].append(len(Hash))

        final_words[group] = common_words
        final_hash[group] = common_hash

    t_count = final_words.pop('tweet_count')
    words_df = pd.DataFrame(final_words)
    words_df = words_df.transpose()
    words_df =  pd.merge(words_df,table.to_frame('node_count'), left_index = True, right_index = True)
    nc = words_df['node_count']
    words_df = words_df.drop('node_count', axis = 1)
    words_df.insert(0, 'node_count', nc)
    words_df.insert(0, 'tweet_count', t_count)
    words_df.insert(0, 'group', words_df.index)


    for key in final_hash:
        while len(final_hash[key]) < 10:
            final_hash[key].append(None)
    h_count = final_hash.pop('hash_count')
    hash_df = pd.DataFrame(final_hash)
    hash_df = hash_df.transpose()
    hash_df = pd.merge(hash_df,table.to_frame('node_count'), left_index = True, right_index = True)
    nc = hash_df['node_count']
    hash_df = hash_df.drop('node_count', axis = 1)
    hash_df.insert(0, 'node_count', nc)
    hash_df.insert(0, 'hash_count', h_count)
    hash_df.insert(0, 'group', hash_df.index)

    if to_csv:
        words_df.to_csv('wordTriage_' + file + '.csv',index = False)
        hash_df.to_csv('hashTriage_' + file + '.csv', index = False)
    else:
        return(words_df, hash_df)
