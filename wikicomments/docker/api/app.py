from flask import Flask, request, jsonify
# Сформируем спискок стоп-слов
from nltk.corpus import stopwords
import pickle
import re
import pandas
import logging
import json

filename_stopwords = '/home/project/app/models/stopwords.txt'
filename_doc2vec = '/home/project/app/models/doc2vec'
filename_model = '/home/project/app/models/model.pkl'

app = Flask(__name__)

with open (filename_stopwords, 'rb') as fp:
		    mystopwords = pickle.load(fp)

with open (filename_doc2vec, 'rb') as fp:
    doc2vec = pickle.load(fp)

with open (filename_model, 'rb') as fp:
    model = pickle.load(fp)

# Токенизация
regex = re.compile("['A-Za-z\-]+")
def tokenize(text, regex=regex, stopwords=mystopwords):
    """ Tokenize all tokens from text string
        Returns array of tokens
    """
    try:
        clean_text = " ".join(regex.findall(text)).lower()
        tokens = [token for token in clean_text.split(' ') if not token in stopwords]
        return tokens
    except:
        return []


@app.route('/predict', methods=['POST'])
def predict():
    """ Api method for prediction value of class
        0 - normal comment
        1 - bad comment
    """
    if request.method == 'POST':

        logging.warning(json.loads(request.data.decode('utf-8'))['text'])

        tokens = tokenize(json.loads(request.data.decode('utf-8'))['text'])
        logging.warning(tokens)

        coef = doc2vec.infer_vector(tokens, steps=20)
        result = model.predict([coef])[0]
        
        return jsonify({"class": str(result)})


