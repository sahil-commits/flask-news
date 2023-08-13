from transformers import pipeline
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sn

from sklearn.metrics import roc_auc_score,f1_score,confusion_matrix
from sklearn.model_selection import train_test_split
import pickle


MODEL = "jy46604790/Fake-News-Bert-Detect"
classifier = pipeline("text-classification", model=MODEL, tokenizer=MODEL)

with open('modeltrans.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(_name_)

@app.route('/',methods = ['GET'])
def index():
    return jsonify({'message': 'Hello, World!'})


@app.route('/predict', methods=['POST'])
def predict():
     if request.method == 'POST':
        data = request.form['news_text']
        print(data)
        new_data = ["jaisi krnin wese bari"]
        predictions = model.predict(new_data)
        return jsonify({'message': 'result'+str(predictions)})

if _name_ == '_main_':
    app.run()