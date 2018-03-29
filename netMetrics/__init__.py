def get_hash(tweet):
    """
    Returns list of hashtags in a tweet.  If no hashtags,
    it returns an empty list.

    """
    ht = []
    for h in tweet['entities']['hashtags']:
            ht.append(h['text'])
    return (ht)
