import nltk, string, re
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

# TODO: save stopwords to a document to minimize dependency
stopwords_en = stopwords.words('english')
stopwords_id = stopwords.words('indonesian')
swtw = ['yg', 'com', 'tuh', 'USERNAME', 'username', 'name', 'serambinewscom', 'provider', '%']
stopwords_id = set(stopwords_id + stopwords_en + swtw)
stopwords_id

def rem_url(txt):
  """Remove URLs from a sample string"""
  U = re.sub(r"http\S+", "", txt)
  return U
  
def rem_num(txt):
    """Remove numbers"""
    s = re.sub(r"\d+", "", txt)
    return s

def tokenize(txt):
  """Tokenize string"""
  tokens = word_tokenize(txt)
  return tokens

def rem_punc(txt):
    """Remove punctuation marks"""
    W = re.sub(r'[^\w\s]', '', txt)
    # W = txt.translate(str.maketrans('','',string.punctuation)).lower()
    return W

def rem_stop(txt):
    """Remove stop words"""
    rem = []
    for t in txt:
        if t not in stopwords_id:
            rem.append(t)
    return rem

def stemm(txt):
    """Stemming in Bahasa Indonesia"""
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    K = stemmer.stem(txt)
    return K

def freqs(txt):
    """Determine unique word's frequencies
        Output: tuple
    """
    F = nltk.FreqDist(txt)
    return F.most_common()

# ?????????
def normalisasi(txt):
    """Text normalization or Pre-processing"""
    txt = str(txt)
    txt = stemm(txt)
    U = rem_url(txt)
    R = rem_punc(U)
    N = rem_num(R)
    K = stemm(N)
    T = tokenize(K)
    S = rem_stop(T)
    return S
