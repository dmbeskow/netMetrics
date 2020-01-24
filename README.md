# netMetrics: Automate network based feature extraction

This package was primarily designed to extract Bot-Hunter Tier 3 Features from Twitter Data.  These methods are introduced in

**Beskow, David M., and Kathleen M. Carley.** *Bot conversations are different: leveraging network metrics for bot detection in Twitter.* 2018 IEEE/ACM International Conference on Advances in Social Networks Analysis and Mining (ASONAM). IEEE, 2018. [link](https://ieeexplore.ieee.org/document/8508322)

The methods scrape and extract features for large ego-networks around a target account.



## Installation

```bash
pip install --user --upgrade git+git://github.com/dmbeskow/netMetrics.git
```


## Tier 3 feature extraction

The netMetrics will create features based on a list of user ids.  In order to expedite, make sure your list does not contain duplicates.  

In order to get Tier 3 features for a list of Twitter user ID's, use the following syntax:

```python
import twitter_col
import tweepy
import netMetrics
from pathlib import Path

# Replace the API_KEY and API_SECRET with your application's key and secret.
auth = tweepy.AppAuthHandler( keys['consumer_key'], keys['consumer_secret'])

api = tweepy.API(auth, wait_on_rate_limit=True,
                                   wait_on_rate_limit_notify=True)

if (not api):
    print ("Can't Authenticate")
    sys.exit(-1)

# Get metrics.  Will create and append to CSV
my_ids = # your list of user ids
bot_model = 'path/to/bot_model.pkl'
get_metrics_listOfIDs(list_of_user_ids, api, directory = 'timelines', bot_model,
                          file_prefix = 'twitter_network_metrics_',
                          RS = 777):
```

## Network and Content Exploratory Data Analysis or *triage*

There are two functions in the netMetrics package that are designed to help *triage* a conversation.  In other words, they're designed to help the analyst determine what kind of topics are in communities or groups.  

The first function is able to take a Twitter stream saved into a JSON file, create a conversational network (links = mention, retweet, reply), compute the communities or goups in this network with Louvaine clustering algorithm, and then identify top words and hashtags for each louvaine community.  

The use of this code is illustrated below:

```python
from netMetrics import triage
words_df, hash_df = triage.network_triage('tweets.json.gz', to_csv = False, languages = 'en')

```

This can similarly be done for languages as follows:

```python
from netMetrics import triage
words_df, hash_df = triage.word_triage_by_language('tweets.json.gz', to_csv = False, languages = 'en')
```

The `triage` module will require `nltk`.  Additionally, if you haven't done so, you will need to run the following code:

```python
import nltk
nltk.download('stopwords')
nltk.download('punkt')
```
