import pandas as pd
import translatepy

en = pd.read_csv('tweets.csv', usecols=['tweet'])
le = en.values.tolist()
translator = translatepy.Translator()

ti = []
for x in le:
    id = translator.translate(x, "id")
    ti.append(id)

un = pd.read_csv('tweets.csv', usecols=['username'])
nm = un.values.tolist()

tf = pd.DataFrame(ti)
df = pd.DataFrame()
df['username'] = nm
df['tweet_og'] = le
df['tweet_id'] = tf
df.to_csv('translated_tweets.csv')
