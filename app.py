import os, sys, shutil, time

from flask import Flask, request, jsonify, render_template,send_from_directory
import pandas as pd
from sklearn.externals import joblib
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import urllib.request
import json



app = Flask(__name__)



@app.route('/')
def root():
    return render_template('index.html')



@app.route('/about.html')
def about():
    return render_template('about.html')

@app.route('/contact.html')
def contact():
    return render_template('contact.html')

@app.route('/result.html', methods = ['POST'])
def predict():
    tree = joblib.load('model/tree')
    print('model loaded')

    if request.method == 'POST':
        gl = request.form['Glucose']
        bp = request.form['BloodPressure']
        inm = request.form['INSULIN']
        dpf= request.form['DIABETES_FUNCTION']
        bmi = request.form['BMI']
        age= request.form['AGE']
        data=np.array([gl,bp,inm,dpf,bmi,age])
        
        db=data.reshape(1,-1)
        print(db)
        my_prediction = tree.predict(db)
        if my_prediction[0] == 1:
            my_prediction ='''You're Diabetes Patient!!!!!!!!'''
        elif my_prediction[0] == 0:
            my_prediction ='''DIABETES is not detected!!!!!!!'''



    return render_template('result.html', prediction = my_prediction)


if __name__ == '__main__':
    app.run(debug = True)
