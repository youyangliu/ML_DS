# coding: utf-8

# In[7]:

# --------------------------------import libraries-------------------------------
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.utils import shuffle
from scipy.sparse import hstack, csr_matrix, diags
from scipy.stats import itemfreq
from nltk.stem import PorterStemmer
import random

# In[197]:

# --------------------------------load data-------------------------------
data = pd.read_csv("reviews_tr.csv")
data = shuffle(data, random_state=0, n_samples=int(len(data) / 2))
test_data = pd.read_csv("reviews_te.csv")


# In[186]:

# --------------------------------build online perceptron function-------------------------------
def online_perceptron(train_data, train_label):
    '''
    set parameters
    train_data: scipy sparse matrix, feature matrix that contains processed frequency counts of
                input texts.
    train_label: np.array, corresponding traning labels
    '''
    w = np.zeros((np.shape(train_data)[1], 1))

    random.seed(0)
    # shuffle training data between each pass
    sel = random.sample(range(len(train_label)), len(train_label))
    train_s = train_data[sel]
    labels_s = train_label[sel]

    for i in range(0, len(train_label)):
        y_t = labels_s[i]
        x_t = train_s[i, :]
        if y_t * (x_t.dot(w)) <= 0:
            w = w + y_t * x_t.transpose()

    # shuffle training data between each pass
    sel = random.sample(range(len(train_label)), len(train_label))
    train_s = train_data[sel]
    labels_s = train_label[sel]

    w_f = w
    count = 1

    for j in range(0, len(train_label)):
        y_t = labels_s[j]
        x_t = train_s[j, :]
        if y_t * (x_t.dot(w)) <= 0:
            w = w + y_t * x_t.transpose()
        w_f += w
        count += 1

    w_final = (1 / (len(train_label) + 1)) * w_f

    return w_final


# In[187]:

# --------------------------------build evaluate function-------------------------------
def evaluate(data_name, coef, train_data, train_label, test_data, test_label):
    '''
    set parameters
    data_name: data representation name
    coef: numpy column vector returned by online_perception function
    train_data: scipy sparse matrix, feature matrix that contains processed frequency counts of
                training texts.
    train_label: numpy column vector, corresponding traning labels
    test_data: scipy sparse matrix, feature matrix that contains processed frequency counts of
               testing texts.
    test_label: numpy column vector, corresponding test labels
    '''
    y_hat_tr = train_data.dot(coef)
    np.place(y_hat_tr, y_hat_tr <= 0, -1)
    np.place(y_hat_tr, y_hat_tr > 0, 1)
    tr_error = 1 - sum(y_hat_tr == train_label) / len(train_label)

    y_hat_te = test_data.dot(coef)
    np.place(y_hat_te, y_hat_te <= 0, -1)
    np.place(y_hat_te, y_hat_te > 0, 1)
    te_error = 1 - sum(y_hat_te == test_label) / len(test_label)

    print(data_name, "\n"
                     "Training error: ", tr_error, "\n"
                                                   "Testing error: ", te_error)


# ## Unigram

# In[198]:

# generate unigram representation, including words consisting of only 1 character
uni_vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
uni = uni_vectorizer.fit_transform(data['text'])
# add intercept
inter_tr = csr_matrix(np.ones((np.shape(uni)[0], 1)))
uni = hstack([inter_tr, uni], format='csr')

# In[200]:

# replace 0 values in the label with -1
uni_labels = np.array(data['label'])
np.place(uni_labels, uni_labels == 0, -1)

# In[201]:

# transform test data to unigram representation
uni_test = uni_vectorizer.transform(test_data['text'])
inter_te = csr_matrix(np.ones((np.shape(uni_test)[0], 1)))
uni_test = hstack([inter_te, uni_test], format='csr')

# In[202]:

# reshape label arrays for later use
train_labels = uni_labels.reshape((len(uni_labels), 1))
test_labels = test_data['label'].values.reshape((np.shape(uni_test)[0], 1))
np.place(test_labels, test_labels == 0, -1)

# implement online perceptron algorithm on the unigram representation
uni_coef = online_perceptron(uni, uni_labels)
# evaluate results
evaluate("Unigram", uni_coef, uni, train_labels, uni_test, test_labels)

# In[203]:

# find top 10 words with highest weights
uni_coef_r = uni_coef[1:].reshape((1, len(uni_coef) - 1)).tolist()[0]


def f(a, N, ord):
    return np.argsort(a)[::ord][:N]


high_w = {}
for i in f(uni_coef_r, 10, -1):
    high_w[uni_vectorizer.get_feature_names()[i]] = uni_coef_r[i]

for k in sorted(high_w):
    print(k, ": ", high_w[k])

# In[204]:

# find top 10 words with lowest weights
low_w = {}
for i in f(uni_coef_r, 10, 1):
    low_w[uni_vectorizer.get_feature_names()[i]] = uni_coef_r[i]

for k in sorted(low_w):
    print(k, ": ", low_w[k])

# ## Bigram

# In[185]:

# generate bigram representation, including words consisting of only 1 character
bi_vectorizer = CountVectorizer(ngram_range=(2, 2), token_pattern=r'\b\w+\b')
bi = bi_vectorizer.fit_transform(data['text'])
# add intercept
inter_tr = csr_matrix(np.ones((np.shape(bi)[0], 1)))
bi = hstack([inter_tr, bi], format='csr')

# tranform test data into bigram representation
bi_test = bi_vectorizer.transform(test_data['text'])
inter_te = csr_matrix(np.ones((np.shape(bi_test)[0], 1)))
bi_test = hstack([inter_te, bi_test], format='csr')

# replace 0 values in the label with -1
bi_labels = np.array(data['label'])
np.place(bi_labels, bi_labels == 0, -1)

# In[188]:

# reshape label arrays for later use
train_labels = bi_labels.reshape((len(bi_labels), 1))
test_labels = test_data['label'].values.reshape((np.shape(bi_test)[0], 1))
np.place(test_labels, test_labels == 0, -1)

# implement online percpetron algorithm on bigram representation
bi_coef = online_perceptron(bi, bi_labels)
# evaluate results
evaluate("Bigram", bi_coef, bi, train_labels, bi_test, test_labels)

# ## Tf-idf

# In[205]:

# generate term frequency representation, including words consisting of only 1 character
tf_vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
tf = tf_vectorizer.fit_transform(data['text'])

# In[206]:

# calculate df
word_count_by_col = itemfreq(tf.nonzero()[1]).transpose()
# calculate idf
log_idf = np.log10(np.shape(tf)[0] / (word_count_by_col[1]))
idf_mtx = diags(log_idf, 0)
# calculate tf-idf
tfidf = tf.dot(idf_mtx)

# add intercept
inter_tr = csr_matrix(np.ones((np.shape(tfidf)[0], 1)))
tfidf = hstack([inter_tr, tfidf], format='csr')

# In[207]:

# transform test data into term frequency representation
tf_test = tf_vectorizer.transform(test_data['text'])
# calculate tf-idf on test data
tfidf_test = tf_test.dot(idf_mtx)

# add intercept
inter_te = csr_matrix(np.ones((np.shape(tfidf_test)[0], 1)))
tfidf_test = hstack([inter_te, tfidf_test], format='csr')

# In[208]:

# replace 0 values in the label with -1
tfidf_labels = np.array(data['label'])
np.place(tfidf_labels, tfidf_labels == 0, -1)

# implement online perceptron algorithm on tf-idf representation
tfidf_coef = online_perceptron(tfidf, tfidf_labels)
# evaluate results
evaluate("Tf-idf", tfidf_coef, tfidf, train_labels, tfidf_test, test_labels)


# ## Stemming + Unigram

# In[46]:

# define stemming function
def pstem(row):
    '''
    row: takes a row from a dataframe
    '''
    ps = PorterStemmer()

    stem = ""
    for w in (row['text'].split()):
        stem = ' '.join((stem, ps.stem(w)))
    return stem


# In[9]:

# create a new column on training data containing stemmed text using pstem function
data['stemmed_text'] = data.apply(lambda row: pstem(row), axis=1)

# In[11]:

# save data for repeated use
data.to_csv("reviews_tr_50_stemmed.csv")

# In[18]:

# create a new column on test data containing stemmed text using pstem function
test_data['stemmed_text'] = test_data.apply(lambda row: pstem(row), axis=1)

# In[20]:

# save test data for repeated use
test_data.to_csv("reviews_te_stemmed.csv")

# In[210]:

# generate unigram representation on stemmed text, including words consisting of only 1 character
stm_vectorizer = CountVectorizer(token_pattern=r'\b\w+\b')
stm = stm_vectorizer.fit_transform(data['stemmed_text'])
# add intercept
inter_tr = csr_matrix(np.ones((np.shape(stm)[0], 1)))
stm = hstack([inter_tr, stm], format='csr')

# In[211]:

# replace 0 values in the label with -1
stm_labels = np.array(data['label'])
np.place(stm_labels, stm_labels == 0, -1)

# In[212]:

# tranform stemmed test data into unigram representation
stm_test = stm_vectorizer.transform(test_data['stemmed_text'])
inter_te = csr_matrix(np.ones((np.shape(stm_test)[0], 1)))
stm_test = hstack([inter_te, stm_test], format='csr')

# In[213]:

# reshape label arrays for later use
train_labels = stm_labels.reshape((len(stm_labels), 1))
test_labels = test_data['label'].values.reshape((np.shape(stm_test)[0], 1))
np.place(test_labels, test_labels == 0, -1)

# implememnt online perceptron algorithm on stemming + unigram representation
stm_coef = online_perceptron(stm, stm_labels)
# evaluate results
evaluate("Stemming + Unigram", stm_coef, stm, train_labels, stm_test, test_labels)
