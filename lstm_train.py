# -*- coding: utf-8 -*-
import re

import pandas as pd
import numpy as np 
import jieba
import multiprocessing
from keras import utils as np_utils
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.layers.core import Dense, Dropout,Activation
np.random.seed(1337)
import sys
sys.setrecursionlimit(1000000)
import yaml

cpu_count = multiprocessing.cpu_count()
vocab_dim = 100
n_iterations = 5 #
n_exposures = 10 # 所有频数超过10的词语
window_size = 5
n_epoch = 7
input_length = 100
maxlen = 100
batch_size = 128

def loadfile():
    neg=pd.read_csv('data/negt.csv',header=None,index_col=None)
    pos=pd.read_csv('data/post.csv',header=None,index_col=None,error_bad_lines=False)
    neu=pd.read_csv('data/neutralt.csv', header=None, index_col=None)

    combined = np.concatenate((pos[0], neu[0], neg[0]))
    y = np.concatenate((np.ones(len(pos), dtype=int), np.zeros(len(neu), dtype=int), 
                        -1*np.ones(len(neg),dtype=int)))
    return combined,y

#对句子经行分词，并去掉换行符
def tokenizer(text):
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    text = re.sub('[A-Za-z0-9]|/d+', '', text)
    text = re.sub('[~,@,。,[,:,/,!,#,…,：,！,，,？,-,<<, ,【,；,】，《,》，～，]', '', text)
    text = re.sub(r, '', text)
    text = [jieba.lcut(document.replace('\n',''))for document in text]
    return text



def word2vec_train(combined):
    model = Word2Vec(size=vocab_dim,  #vocab_dim = 100
                     min_count=n_exposures,  #n_exposures = 10
                     window=window_size,   #window_size = 7
                     workers=cpu_count,
                     iter=n_iterations)   #n_iterations = 5
    model.build_vocab(combined) # input: list
    model.train(combined,epochs=model.iter,total_examples=model.corpus_count)
    model.save('model/Word2vec_model.pkl')
    index_dict, word_vectors,combined = create_dictionaries(model=model,combined=combined)
    return  index_dict, word_vectors,combined


def create_dictionaries(model=None,combined=None):

    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}
        w2vec = {word: model[word] for word in w2indx.keys()}

        def parse_dataset(combined): # 闭包-->临时使用
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) 
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')

def get_data(index_dict,word_vectors,combined,y):

    n_symbols = len(index_dict) + 1  
    embedding_weights = np.zeros((n_symbols, vocab_dim)) 
    for word, index in index_dict.items(): 
        embedding_weights[index, :] = word_vectors[word]
    x_train, x_test, y_train, y_test = train_test_split(combined, y, test_size=0.1)
    y_train = np_utils.to_categorical(y_train,num_classes=3)
    y_test = np_utils.to_categorical(y_test,num_classes=3)
    return n_symbols,embedding_weights,x_train,y_train,x_test,y_test

def train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test):
    print ('Defining Model...')
    model = Sequential()
    model.add(Embedding(output_dim=vocab_dim,
                        input_dim=n_symbols,
                        mask_zero=True,
                        weights=[embedding_weights],
                        input_length=input_length))

    model.add(LSTM(output_dim=100, activation='tanh'))
    model.add(Dropout(0.5))
    model.add(Dense(3, activation='softmax'))
    model.add(Activation('softmax'))
    '''
    model.add(Conv1D(128,4,padding='valid', activation='relu', strides=1))
    model.add(Conv1D(64,4, padding='valid', activation='relu', strides=1))
    model.add(Conv1D(32,4, padding='valid', activation='relu', strides=1))
    model.add(MaxPooling1D(pool_size=pool_size))
    model.add(Dropout(0.5))
    model.add(LSTM(50))
    model.add(Dropout(0.1))
    model.add(Dense(3, activation='softmax'))
    model.add(Activation('softmax'))
    '''
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    print (model.summary())
    model.fit(x_train, y_train,validation_data=(x_test,y_test),batch_size=batch_size, epochs=n_epoch,verbose=1)
    print ("Evaluate...")
    score = model.evaluate(x_test, y_test,batch_size=batch_size,verbose=1)#  loss,acc
    yaml_string = model.to_yaml()
    with open('model/lstm.yml', 'w') as outfile:
        outfile.write(yaml.dump(yaml_string, default_flow_style=True))
    model.save_weights('model/lstm.h5')
    print ('Test score:', score)

print ('Loading Data...')
combined,y=loadfile()
print (len(combined),len(y))
print ('Tokenising...')
combined = tokenizer(combined)
print ('Training a Word2vec model...')
index_dict, word_vectors,combined=word2vec_train(combined)

n_symbols,embedding_weights,x_train,y_train,x_test,y_test=get_data(index_dict, word_vectors,combined,y)
print (x_train.shape,y_train.shape)
train_lstm(n_symbols,embedding_weights,x_train,y_train,x_test,y_test)




