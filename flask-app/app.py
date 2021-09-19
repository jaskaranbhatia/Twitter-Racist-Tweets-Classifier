from flask import Flask,render_template,url_for,request
import pandas as pd 
import numpy as np
import pickle
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split 
import joblib
import math

app = Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
        train = pd.read_csv('train.csv')
        df= pd.read_csv("clean_dataset.csv")
        cv = CountVectorizer(max_df=0.90,min_df=2,max_features=1000,stop_words='english')
        cv.fit_transform(df['tidy_tweet'].values.astype('U')) 
        bow = cv.fit_transform(df['tidy_tweet'].values.astype('U'))
        X = bow[:31962,:]
        x_train,x_test,y_train,y_test = train_test_split(X,train['label'],test_size=0.3)
        lg = LogisticRegression()
        lg.fit(x_train,y_train)
        if request.method == 'POST':
            message = request.form['message']
            data = [message]
            vect = cv.transform(data).toarray()
            my_prediction = lg.predict(vect)
            result = lg.predict_proba(vect)
            result_precise = math.trunc(result[0][1]*100)
        return render_template('result.html',prediction = my_prediction, tweet = message, res = result_precise)



if __name__ == '__main__':
	app.run()