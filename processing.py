import nltk, re
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

#TODO: save stopwords to a document to minimize dependency
stopwords_en = stopwords.words("english")
stopwords_id = stopwords.words("indonesian")
swtw = [
    "yg",
    "com",
    "tuh",
    "USERNAME",
    "username",
    "name",
    "%",
]
stopwords_id = set(stopwords_id + stopwords_en + swtw)


def rem_stop(txt):
    """Remove stop words"""
    return [t for t in txt if t not in stopwords_id]


def normalisasi(txt):
    """Text normalization or Pre-processing"""
    # * make sure it's string
    txt = str(txt)
    # * stemming
    factory = StemmerFactory()
    stemmer = factory.create_stemmer()
    txt = stemmer.stem(txt)
    # * remove urls
    txt = re.sub(r"http\S+", "", txt)
    # * remove punctuations
    txt = re.sub(r"[^\w\s]", "", txt)
    # * remove numbers
    txt = re.sub(r"\d+", "", txt)
    # * tokenize string
    txt = word_tokenize(txt)
    # * remove stop words, return normalized strings in a list
    return rem_stop(txt)


def freqs(txt):
    """Determine unique word's frequencies
    Output: tuple
    """
    return nltk.FreqDist(txt).most_common()
