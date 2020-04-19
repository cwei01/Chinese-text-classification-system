import re

import keras
from flask import Flask, request, render_template, jsonify
# from pandas import json
import jieba
import numpy as np
from gensim.models.word2vec import Word2Vec
from gensim.corpora.dictionary import Dictionary
from keras.preprocessing import sequence
import sys

import yaml
from keras.models import model_from_yaml
#keras.backend.clear_session()
from flask import send_from_directory
import warnings
import json, demjson

warnings.filterwarnings("ignore")

np.random.seed(1337)  # For Reproducibility
sys.setrecursionlimit(1000000)

# define parameters
maxlen = 100

app = Flask(__name__)


@app.route('/will_predict', methods=['POST'])
def will_predict1():
    receive = request.values['message']
    #receive=lstm_predict(receive)
    print('will_predict : ' + receive)
    return jsonify(receive)


@app.route('/predict1', methods=['POST'])
def predict1():
    keras.backend.clear_session()
    receive = request.values['message']
    res = lstm_predict(receive)
    print('predict1 : ' + res)
    return jsonify(res)


@app.route('/predict2', methods=['POST'])
def predict2():
    #print('fdf')
    keras.backend.clear_session()
    data1 = request.values['message1']
    data2 = request.values['message2']
    model = Word2Vec.load('model/Word2vec_model.pkl')
    po = model.similarity(data1, data2)
    print(po)
    return jsonify(str(po))


@app.route('/predict3', methods=['POST'])  # 添加路由：根
def predict3():
    keras.backend.clear_session()
    data = request.values['message']
    model = Word2Vec.load('model/Word2vec_model.pkl')
    sim5 = model.most_similar(data, topn=5)
    sim5=list(sim5)
    #print(sim5)
    # print(u' 与相关的词有: \n')
    #sim=""
    for key in sim5:
      print(key[0])
      #sim=sim+"".join(key)
    #sim5=sim5+"\n"
    return jsonify(( sim5[0].__str__()+"</br>"
                    +sim5[1].__str__()+'</br>'
                    +sim5[2].__str__()+'</br>'
                    +sim5[3].__str__()+'</br>'
                    +sim5[4].__str__()+'</br>'))


@app.route('/', methods=["GET"])
def index():
    return render_template("index.html")


@app.route('/css/<path:path>')
def send_css(path):
    return send_from_directory('css', path)


@app.route('/data/<path:path>')
def send_data(path):
    return send_from_directory('data', path)


@app.route('/js/<path:path>')
def send_js(path):
    return send_from_directory('js', path)


@app.route('/imgs/<path:path>')
def send_imgs(path):
    return send_from_directory('imgs', path)


@app.route('/fonts/<path:path>')
def send_fonts(path):
    return send_from_directory('fonts', path)


@app.route('/images/<path:path>')
def send_images(path):
    return send_from_directory('images', path)


@app.route('/lstm_data/<path:path>')
def send_lstm_data(path):
    return send_from_directory('lstm_data', path)


@app.route('/model/<path:path>')
def send_model(path):
    return send_from_directory('model', path)


def create_dictionaries(model=None,
                        combined=None):

    if (combined is not None) and (model is not None):
        gensim_dict = Dictionary()
        gensim_dict.doc2bow(model.wv.vocab.keys(),
                            allow_update=True)
        #  freqxiao10->0 所以k+1
        w2indx = {v: k + 1 for k, v in gensim_dict.items()} 
        w2vec = {word: model[word] for word in w2indx.keys()} 

        def parse_dataset(combined):
            data = []
            for sentence in combined:
                new_txt = []
                for word in sentence:
                    try:
                        new_txt.append(w2indx[word])
                    except:
                        new_txt.append(0)  
                data.append(new_txt)
            return data  

        combined = parse_dataset(combined)
        combined = sequence.pad_sequences(combined, maxlen=maxlen)  
        return w2indx, w2vec, combined
    else:
        print('No data provided...')


def input_transform(string):
    r = '[’!"#$%&\'()*+,-./:;<=>?@[\\]^_`{|}~]+'
    string = re.sub('[A-Za-z0-9]|/d+', '', string)
    string = re.sub('[~,@,。,[,:,/,!,#,…,：,！,，,？,-,<<, ,【,；,】，《,》，～，]', '', string)
    string = re.sub(r, '', string)
    words = jieba.lcut(string)
    print(words)
    words = np.array(words).reshape(1, -1)
    model = Word2Vec.load('model/Word2vec_model.pkl')
    _, _, combined = create_dictionaries(model, words)
    y1 = model.similarity(u'北京', u'上海')
    #print(y1)
    return combined


def lstm_predict(string):
    with open('model/lstm.yml', 'r') as f:
        yaml_string = yaml.load(f)
    model = model_from_yaml(yaml_string)
    model.load_weights('model/lstm.h5')
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    data = input_transform(string)
    data.reshape(1, -1)
    result = model.predict_classes(data)
    # print result # [[1]]
    if result[0] == 1:
        s = '+1'
        # print (string,' positive')
        return s

    elif result[0] == 0:
        s = '0'
        # print (string,' neural')
        return s
    else:
        s = '-1'
        return s
        # print(string, end=' ')
        # while lenth > 0:
        # lenth = lenth - 1
        # print('*', end='')


if __name__ == '__main__':
    app.run(debug=True)
    
