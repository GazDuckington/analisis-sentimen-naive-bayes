# %%
import pandas as pd
import re, sys
sys.path.insert(0, '/home/gaz/dev/proyek-sk/') 
from processing import rem_url, normalisasi, rem_num
# %%
def remove_URL(sample):
    """Remove URLs from a sample string"""
    return re.sub(r"http\S+", "", sample)

def senti(t):
    if t == 'negative':
        return 0
    elif t == 'positive':
        return 1
# %%
tweet = pd.read_csv("/home/gaz/dev/proyek-sk/dataset/all_tweets_raw.csv")
tweet['Text'] = tweet['Text'].apply(lambda x: normalisasi(str(x)))
tweet['Sentiment'] = tweet['Sentiment'].apply(lambda x: senti(x))
tweet.to_csv("/home/gaz/dev/proyek-sk/dataset/all_tweets.csv")
# %%
at = pd.read_csv("/home/gaz/dev/proyek-sk/dataset/all_tweets.csv")
an = at[at['Sentiment'] == 0]
an.to_csv("/home/gaz/dev/proyek-sk/dataset/all_neg_tweets.csv")
ap = at[at['Sentiment'] == 1]
ap.to_csv("/home/gaz/dev/proyek-sk/dataset/all_pos_tweets.csv")
# %%
dataset = pd.read_csv("/home/gaz/dev/proyek-sk/dataset/gnews.csv")
# dataset['content'].replace('', np.nan, inplace=True)
dataset.dropna(subset=['content'], inplace=True)
dataset['content'] = dataset['content'].apply(lambda x: normalisasi(x))
# dataset['content'] = dataset['content'].str.replace('\d+', '')
dataset.to_csv('/home/gaz/dev/proyek-sk/dataset/news.csv')
dataset['content']
# %%
# tweet_p = pd.read_csv("/home/gaz/dev/proyek-sk/dataset/all_pos_tweets.csv")
# tweet_p['Text'] = tweet_p['Text'].apply(lambda x: normalisasi(x))
# tweet_p['Text'] = tweet_p['Text'].apply(lambda x: ' '.join(x))

# tweet_n = pd.read_csv("/home/gaz/dev/proyek-sk/dataset/all_neg_tweets.csv")
# tweet_n['Text'] = tweet_n['Text'].apply(lambda x: normalisasi(x))
# tweet_n['Text'] = tweet_n['Text'].apply(lambda x: ' '.join(x))

# unik_p = pd.DataFrame()
# unik_n = pd.DataFrame()

# unik_n['neg'] = tweet_n['Text'].str.lower().str.findall("\w+").sum()
# unik_p['pos'] = tweet_p['Text'].str.lower().str.findall("\w+").sum()

# outp = pd.DataFrame()
# t = unik_n['neg'].value_counts()
# outp['w'] = t.keys()
# outp['c'] = t.values
# outp.to_csv('kataunikneg.csv')