import numpy as np

import pandas as pd

import nltk
from nltk.corpus import stopwords

from string import punctuation

import pymorphy2

import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV
from lem import lem
from keras.models import Model
from tensorflow.keras.models import Sequential
from keras.layers import Input, Dense, Embedding, SpatialDropout1D, concatenate
from keras.layers import GRU, Bidirectional, GlobalAveragePooling1D, GlobalMaxPooling1D
from keras.preprocessing import text, sequence
from keras.callbacks import Callback
from keras.utils.data_utils import pad_sequences
import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
import pickle
import seaborn as sns

def predict(text):
    model = joblib.load("model.pkl")
    with open('tokenizer.pickle', 'rb') as handle:
        loaded_tokenizer = pickle.load(handle)

    texts = lem(str(text))
    print(texts)
    max_features = 30000
    maxlen = 300

    token_texts = loaded_tokenizer.texts_to_sequences(texts)
    texts_seq = pad_sequences(token_texts, maxlen=maxlen)

    y_pred_pred = model.predict(texts_seq, verbose=1, batch_size=2)

    print(round(y_pred_pred[0][0],4))
print(predict('Влад лох, сосет хуй'))
