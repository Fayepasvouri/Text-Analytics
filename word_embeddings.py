"""
Faye

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

import re

# Importing required libraries
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords

# keras
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

from keras.layers import Embedding

from keras.layers import Dense, Input, GlobalMaxPooling1D
from keras.layers import Conv1D, MaxPooling1D, Embedding
from keras.models import Model
from keras.layers import Input, Dense, Embedding, Conv2D, MaxPooling2D, Dropout,concatenate
from keras.layers.core import Reshape, Flatten
from keras.callbacks import EarlyStopping
from keras.optimizers import SGD, RMSprop, Adagrad, Adadelta, Adam, Adamax, Nadam
from keras.models import Model
from keras import regularizers

# gensim
import gensim
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess

from gensim.models.keyedvectors import KeyedVectors

import gensim.downloader as api

api.info("text8")  # retrieve information about text8 dataset
{u'checksum': u'68799af40b6bda07dfa47a32612e5364',
 u'description': u'Cleaned small sample from wikipedia',
 u'file_name': u'text8.gz',
 u'parts': 1,
 u'source': u'http://mattmahoney.net/dc/text8.zip'}

api.info()

def cleantxt(txt):
  
    # collecting english stop words from nltk-library
    stpw = stopwords.words('english')
    
    # Adding custom stop-words
    stpw.extend(['www','http','utc'])
    stpw = set(stpw)
    
    # using regex to clean the text
    txt = re.sub(r"\n", " ", txt)
    txt = re.sub("[\<\[].*?[\>\]]", " ", txt)
    txt = txt.lower()
    txt = re.sub(r"[^a-z ]", " ", txt)
    txt = re.sub(r"\b\w{1,3}\b", " ",txt)
    txt = " ".join([x for x in txt.split() if x not in stpw])
    return txt


def load_data():
    """
    Loads data and returns train, val, and test splits
    """
    # Load the train dataset
    df = pd.read_csv("C:/Users/Faye/Desktop/Master/textanalytics.csv")
    print(df)
    print(df.dtypes)
    df1= df.replace(np.nan, '', regex=True)
    print(df1)
    df1['Comments'] = df1['Comments'].astype(str)
    
    # Clean the text
    df1['Comments'] = df1.Comments.apply(lambda x : cleantxt(x))
    
    # separate explanatory and dependent variables
    X = df1.iloc[:,1]
    y = df1.iloc[:,2:]

    # split for cross-validation (train-60%, validation 20% and test 20%)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=123)
    X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.5, random_state=123)

    return X_train, X_val, X_test, y_train, y_val, y_test

X_train, X_val, X_test, y_train, y_val, y_test = load_data()

NUM_WORDS = 20000

# Define/Load Tokenize text function
tokenizer = Tokenizer(num_words=NUM_WORDS,filters='!"#$%&()*+,-./:;<=>?@[\\]^_`{|}~\t\n\'',
                      lower=True)

# Fit the function on the text
tokenizer.fit_on_texts(X_train)

# Count number of unique tokens
word_index = tokenizer.word_index
print('Found %s unique tokens.' % len(word_index))

sequences_train = tokenizer.texts_to_sequences(X_train)
sequences_valid=tokenizer.texts_to_sequences(X_val)

X_train = pad_sequences(sequences_train,maxlen=50)
X_val = pad_sequences(sequences_valid,maxlen=X_train.shape[1])

# Convert target to array
y_train = np.asarray(y_train)
y_val = np.asarray(y_val)

# Printing shape
print('Shape of X train and X validation tensor:', X_train.shape,X_val.shape)
print('Shape of label train and validation tensor:', y_train.shape,y_val.shape)

df = pd.read_csv('C:/Users/Faye/Desktop/Master/textanalytics.csv')
df['Comments'] = df['Comments'].astype(str)
corpus_text = '\n'.join(df[:5000]['Comments'])
sentences = corpus_text.split('\n')
sentences = [line.lower().split(' ') for line in sentences]

def clean(s):
    return [w.strip(',."!?:;()\'') for w in s]
sentences = [clean(s) for s in sentences if len(s) > 0]


model = Word2Vec(sentences, size=100, window=5, min_count=3, workers=4)

vectors = model.wv
del model

print(vectors.most_similar('i'))


pd.options.mode.chained_assignment = None 
import numpy as np
import re
import nltk
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt


data = pd.read_csv('C:/Users/Faye/Desktop/Master/textanalytics.csv')
data['Comments'] = data['Comments'].astype(str)
STOP_WORDS = nltk.corpus.stopwords.words()

def clean_sentence(val):
    "remove chars that are not letters or numbers, downcase, then remove stop words"
    regex = re.compile('([^\s\w]|_)+')
    sentence = regex.sub('', val).lower()
    sentence = sentence.split(" ")
    
    for word in list(sentence):
        if word in STOP_WORDS:
            sentence.remove(word)  
            
    sentence = " ".join(sentence)
    return sentence

def clean_dataframe(data):
    "drop nans, then apply 'clean_sentence' function to question1 and 2"
    data = data.dropna(how="any")
    
    for col in ['Comments']:
        data[col] = data[col].apply(clean_sentence)
    
    return data

data = clean_dataframe(data)
data.head(10)
print(data.head(10))



def build_corpus(data):
    "Creates a list of lists containing words from each sentence"
    corpus = []
    for col in ['Comments']:
        for sentence in data[col].iteritems():
            word_list = sentence[1].split(" ")
            corpus.append(word_list)
            
    return corpus

corpus = build_corpus(data)        
corpus[0:2]

from gensim.models import word2vec
model = word2vec.Word2Vec(corpus, size=100, window=20, min_count=5, workers=4)
model.wv['property']

def tsne_plot(model):
    "Creates and TSNE model and plots it"
    labels = []
    tokens = []

    for word in model.wv.vocab:
        tokens.append(model[word])
        labels.append(word)
    
    tsne_model = TSNE(perplexity=40, n_components=2, init='pca', n_iter=500, random_state=23)
    new_values = tsne_model.fit_transform(tokens)

    x = []
    y = []
    for value in new_values:
        x.append(value[0])
        y.append(value[1])
        
    plt.figure(figsize=(16, 16)) 
    for i in range(len(x)):
        plt.scatter(x[i],y[i])
        plt.annotate(labels[i],
                     xy=(x[i], y[i]),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')
    plt.show()

tsne_plot(model)

model.most_similar('easy')
