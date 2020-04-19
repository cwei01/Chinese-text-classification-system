#! /bin/env python
# -*- coding: utf-8 -*-
import re

import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence

import yaml
from keras.models import model_from_yaml
np.random.seed(1337)  # For Reproducibility
import sys
sys.setrecursionlimit(1000000)

# define parameters
maxlen = 100

def create_dictionaries(model=None,
                        combined=None):

    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k+1 for k, v in gensim_dict.items()}#所有频数超过10的词语的索引,(k->v)=>(v->k)
        w2vec = {word: model[word] for word in w2indx.keys()}#所有频数超过10的词语的词向量, (word->model(word))

        def parse_dataset(combined): # 闭包-->临时使用
            ''' Words become integers
            '''
            data=[]
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0) # freqxiao10->0
                data.append(new_txt)
            return data # word=>index
        combined=parse_dataset(combined)
        combined= sequence.pad_sequences(combined, maxlen=maxlen)#每个句子所含词语对应的索引，所以句子中含有频数小于10的词语，索引为0
        return w2indx, w2vec,combined
    else:
        print ('No data provided...')


def input_transform(string):
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    string= re.sub('[A-Za-z0-9]|/d+', '', string)
    string = re.sub('[~,@,。,[,:,/,!,#,…,：,！,，,？,-,<<, ,【,；,】，《,》，～，]', '', string)
    string = re.sub(r, '', string)
    words=jieba.lcut(string)
    print(words)
    words=np.array(words).reshape(1,-1)
    model=Word2Vec.load('model/Word2vec_model.pkl')
    _,_,combined=create_dictionaries(model,words)
    return combined


def lstm_predict(string):
    with open('model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)
    model.load_weights('model/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',metrics=['accuracy'])
    data=input_transform(string)
    data.reshape(1,-1)
    result=model.predict_classes(data)
    # print result # [[1]]
    if result[0]==1:
        s='positive'
        print (string,' positive')
        return s

    elif result[0]==0:
        s = 'neural'
        print (string,' neural')
        return s
    else:
        s='******'
        return s
        #print(string, end=' ')
        #while lenth > 0:
            #lenth = lenth - 1
            #print('*', end='')


if __name__=='__main__':
   string = input("请输入中文文本：")
   lstm_predict(string)