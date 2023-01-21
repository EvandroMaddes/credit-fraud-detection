import flask
from flask import request, jsonify
import numpy
import json
import sys
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import joblib
from keras.models import load_model
import os


app = flask.Flask(__name__)
app.config["DEBUG"] = True


#file configuration
filename = 'fraud_model_at_epoch_300.h5'
index_page = 'index_new.html'
new_prediction_page = 'new_prediction.html'
new_prediction_page0 = 'new_prediction0.html'
prediction_page = 'predict_result.html'
author_info_page = 'author_info.html'
model_info_page = 'model_info.html'
training_result_page = 'training_result.html'
dataset = 'data/creditcard.csv'

    


# Standardize features by removing the mean and scaling to unit variance.
def load_database():
    data = pd.read_csv(dataset)
    
    scalar = StandardScaler()

    X = data.drop('Class', axis=1)
    y = data.Class
    
    X_train_v, X_test, y_train_v, y_test = train_test_split(X, y, 
                                                    test_size=0.3, random_state=42)
    X_train, X_validate, y_train, y_validate = train_test_split(X_train_v, y_train_v, 
                                                            test_size=0.2, random_state=42)
    X_train = scalar.fit_transform(X_train)
    X_validate = scalar.transform(X_validate)
    X_test = scalar.transform(X_test)              
    return scalar

@app.route('/')
def home():
    return flask.render_template(index_page)

@app.route('/newPrediction')
def new_prediction():
    return flask.render_template(new_prediction_page)

@app.route('/newPrediction0')
def new_prediction0():
    return flask.render_template(new_prediction_page0)
    
@app.route('/api/v0/predict',methods = ['POST'])
def result():
    if request.method == 'POST':
    #caricare dataset e scalare i dati
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        print("[DEBUG]item to predict: " + ''.join(str(item) for item in to_predict_list))
        to_predict_list = list(map(float, to_predict_list))
        to_predict = numpy.array(to_predict_list).reshape(1,30)
        loaded_model = load_model(filename)
        scalar_single = load_database()
        to_predict = scalar_single.transform(to_predict)
        print("[DEBUG] Model loaded" )
        result = loaded_model.predict(to_predict)
        int_result = result[0,0].round()
        print("[DEBUG]Result: " + str(int_result))
        
        if int_result == 0:
            prediction = "Not a FRAUD :)"
        elif int_result == 1:
            prediction = "This is a FRAUD :("
        else:
            prediction == "Somethings went wrong :("
            
        return flask.render_template(prediction_page,prediction=prediction)  
    
@app.route('/authorInfo', methods=['GET'])
def author_info():
    author = 'Evandro Maddes \n'
    description = 'A fraud detection model using a kaggle dataset'
    return flask.render_template(author_info_page, author=author, description = description)
 
@app.route('/modelInfo', methods=['GET'])
def model_info():
    return flask.render_template(model_info_page)
     
@app.route('/trainingResult', methods=['GET'])
def training_result():
    return flask.render_template(training_result_page)
     
    
app.run()