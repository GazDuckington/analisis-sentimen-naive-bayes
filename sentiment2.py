import numpy as np
from processing import normalisasi
import math

# *kamus frekuensi kata
def kamus_freq(teks, label):
    """kamus frekwensi kata dalam dokumen
    input:
        teks: list string
        label: list integer
    output:
        result : dictionary berisi pair & frekwensi kata
        pair : tuple (kata, label) label adalah nilai sentimen
    """
    result = {}
    for y, t in zip(label, teks):
        for word in normalisasi(t):
            pair = (word,y)
            if pair in result:
                result[pair] += 1
            else:
                result[pair] = 1
    return result

# *NBC training
def train_nbc(freqs, train_x, train_y):
    """melatih naive-bayes classifier
    Args:
        freqs ([dict]): [kamus frekuensi kata data training]
        train_y ([array]): [label sentimen data training]

    Returns:
        logprior[integer]: [nilai probabilitas prior]
        loglikelihood[dict]: [kamus probabilitas likelihood setiap kata pada data training]
    """
    loglikelihood = {}
    logprior = 0
    N_pos = N_neg = 0

    unique_words = set([pair[0] for pair in freqs.keys()])
    v = len(unique_words)

    # menghitung jumlah kata negatif (N_neg) dan positif (N_pos)
    for pair in freqs.keys():
        if pair[1] > 0:
            N_pos += freqs[(pair)]
        else:
            N_neg += freqs[(pair)]    

        # jumlah document
        D = train_y.shape[0]
        # jumlah document positif
        D_pos = sum(train_y)
        # jumlah document negatif
        D_neg = D - sum(train_y)

        # kemungkinan nilai sentimen suatu kata
        logprior = np.log(D_pos) - np.log(D_neg)
        
        for word in unique_words:

            # frekuensi positif dan negatif suatu kata
            freq_pos = freqs.get((word, 1), 0)
            freq_neg = freqs.get((word, 0), 0)
            # probabilitas polaritas kata
            pw_pos = (freq_pos + 1) / (N_pos + v)
            pw_neg = (freq_neg + 1) / (N_neg + v)

            # kemungkinan suatu kata dalam suatu dokumen yang bernilai positif/negatif
            loglikelihood[word] = np.log(pw_pos / pw_neg)

        return logprior, loglikelihood

# *prediktor
def predict_nbc(text, logprior, loglikelihood):
    """prediktor naive-bayes
    Args:
        text ([string]): [kalimat yang ingin diketahui sentimennya]
        logprior ([integer]): [probabilitas prior]
        loglikelihood ([dict]): [probabilitas likelihood]

    Returns:
        p[integer]: [nilai probabilitas sentimen]
    """
    word_l = normalisasi(text)
    p = 0
    p += logprior

    for word in word_l:
        if word in loglikelihood:
            p += loglikelihood[word]
    return p

# *test nbc
def test_nbc(test_x, test_y, logprior, loglikelihood):
    """menguji naivve-bayes classifier
    Input:
        test_x: list kata
        test_y: label dari setiap kata
        logprior: logprior
        loglikelihood: suatu dictionary yang berisikan loglikelihoods untuk setiap kata
    Output:
        accuracy: (klasifikasi benar)/(populasi)
    """
    accuracy = 0  # return this properly
    y_hats = []
    for t in test_x:
        # if the prediction is > 0
        if predict_nbc(t, logprior, loglikelihood) > 0:
            # the predicted class is 1
            y_hat_i = 1
        else:
            # otherwise the predicted class is 0
            y_hat_i = 0

        # append the predicted class to the list y_hats
        y_hats.append(y_hat_i)

    # error is the average of the absolute values of the differences between y_hats and test_y
    arr=np.array([y_hats,test_y])
    error =np.sum(np.diff(arr, axis=0))/len(test_y)

    # Accuracy is 1 minus the error
    accuracy = 1-error

    return accuracy