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
import matplotlib.pyplot as plt

def creat_df():
    data_list = []
    with open("dataset.txt", encoding = 'utf-8') as file:
        for line in file:
            labels = line.split()[0]
            text = line[len(labels)+1:].strip()
            labels = labels.split(",")
            mask = [1 if "__label__NORMAL" in labels else 0,
                    1 if "__label__INSULT" in labels else 0,
                    1 if "__label__THREAT" in labels else 0,
                    1 if "__label__OBSCENITY" in labels else 0]
            data_list.append((text, *mask))
    return pd.DataFrame(data_list, columns=["comment", "normal", "insult", "threat", "obscenity"])

def preprocess_text(text):
    text = text.replace("ё", "е")
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', 'URL', text)
    text = re.sub('[^a-zA-Zа-яА-Я]+', ' ', text)
    text = re.sub(' +', ' ', text)
    text = text.lower()
    text = "".join(text)
    return text.strip()

def get_coefs(word, *arr):
    return word, np.asarray(arr, dtype='float32')


def model_evalulate(model_name):
    testres = model_name.evaluate(feature_test, target_test, verbose=0)

    print("Validation set")
    print('loss:', testres[0])
    print("accuracy:", testres[1])


def model_final(learning_rate, dropout_rate):
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix])(inp)
    x = SpatialDropout1D(dropout_rate)(x)
    x = Bidirectional(GRU(80, return_sequences=True))(x)
    avg_pool = GlobalAveragePooling1D()(x)
    max_pool = GlobalMaxPooling1D()(x)
    conc = concatenate([avg_pool, max_pool])
    outp = Dense(4, activation="sigmoid")(conc)
    model = Model(inputs=inp, outputs=outp)

    model.compile(loss='binary_crossentropy',
                  optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
                  metrics=['accuracy', 'AUC', tf.keras.metrics.Recall(), tf.keras.metrics.Precision()])

    return model


def model_run(X, y, batch_size, n_epochs, learning_rate, dropout_rate):
    model = model_final(learning_rate, dropout_rate)
    history = model.fit(X, y,
                        batch_size=batch_size,
                        epochs=n_epochs,
                        validation_split=0.1)

    return (history, model)

df = creat_df()
df = pd.read_csv('lemmatization+preprocessed.csv')
df['comment'] = df['comment'].astype(str)
df = df.drop('Unnamed: 0', axis=1)

df_train, df_test = train_test_split(df, train_size=0.8)
df_train['comment'] = df_train['comment'].apply(preprocess_text).apply(lambda x: preprocess_text(x))
df_test['comment'] = df_test['comment'].apply(preprocess_text).apply(lambda x: preprocess_text(x))
normal_class = df_train[df_train['normal'] > 0].index


part35percent = (len(normal_class)*35)/100
drop_part_normal_class = normal_class[ : round(part35percent)]
df_train = df_train.drop(drop_part_normal_class, axis=0)

feature_train = df_train['comment'].values
feature_test = df_test['comment'].values

target_train = df_train.drop('comment', axis = 1).values
target_test = df_test.drop('comment', axis = 1).values

max_features = 30000
maxlen = 300

tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(feature_train)+list(feature_test))

with open('tokenizer.pickle', 'wb') as handle:
    pickle.dump(tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)

token_feature_train = tokenizer.texts_to_sequences(feature_train)
token_feature_test = tokenizer.texts_to_sequences(feature_test)

feature_train = pad_sequences(token_feature_train, maxlen=maxlen)
feature_test = pad_sequences(token_feature_test, maxlen=maxlen)

EMBEDDING_FILE = 'cc.ru.300.vec'
embed_size = 300

embeddings_index = dict(get_coefs(*o.rstrip().rsplit(' ')) for o in open(EMBEDDING_FILE, encoding='utf-8'))
word_index = tokenizer.word_index
nb_words = min(max_features, len(word_index))
embedding_matrix = np.zeros((nb_words, embed_size))
for word, i in word_index.items():
    if i >= max_features: continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None: embedding_matrix[i] = embedding_vector
top_batch_size = 32
top_epochs = 4
# top_optimizer = Adam
top_learning_rate = 0.01
top_dropout_rate = 0.3

history, model = model_run(feature_train, target_train, top_batch_size, top_epochs, top_learning_rate, top_dropout_rate)
model_evalulate(model)
joblib.dump(model, "model.pkl")

model = joblib.load("model.pkl")
with open('tokenizer.pickle', 'rb') as handle:
    loaded_tokenizer = pickle.load(handle)

texts = ['Если ты не собака , то ты мастурбатор']
texts=lem(str(texts))
print(texts)
max_features = 30000
maxlen = 300

token_texts = loaded_tokenizer.texts_to_sequences(texts)
texts_seq = pad_sequences(token_texts, maxlen=maxlen)

y_pred_pred = model.predict(texts_seq, verbose = 1, batch_size = 2)

for i in y_pred_pred:
    print(texts[0],'\n')
    print("normal", "insult", "threat", "obscenity")
    for j in i:
        print('{:.4f}'.format(j), end=' ')
    print('\n')




