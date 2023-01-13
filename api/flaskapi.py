import joblib
from flask import Flask, request, render_template, jsonify
import json
import requests
import pandas as pd
import uvicorn
from sklearn.model_selection import train_test_split
import warnings
import shap
import numpy as np


app = Flask(__name__)  

with app.app_context():
    # within this block, current_app points to app.
    print(app.name)

def shap_interpretation(train_data, classifier, transformer, test_data, features_names, train_sample_size = 5000):
  train_data_sampled = shap.sample(train_data, train_sample_size, random_state=0)
  explainer = shap.KernelExplainer(classifier.predict_proba, train_data_sampled)
  # explainer = shap.TreeExplainer(classifier)
  reshaped_test_data = test_data.reshape(1,-1)
  shap_values = explainer.shap_values(reshaped_test_data)
  expected_value = explainer.expected_value
  return {"shap_values" : shap_values, "expected_value" : expected_value}

@app.route('/predict/<client>', methods = ["POST"])
def score_calculation(client):
    #client = request.form["client"]
    client_id = int(client)
    df = pd.read_csv(r"static/test_set_2.csv")
    df = df[df["SK_ID_CURR"]==client_id]
    X = df.drop(columns = ["TARGET","SK_ID_CURR"]).values
    try:
        f=open('static/probas.json')
        probas = json.load(f)
        probability = probas[client]
    except:
        warnings.filterwarnings("ignore")
        model = joblib.load('static/best_model.pkl')
        prediction = model.predict(X)[0]
        probability = model.predict_proba(X)[0][1]
    if X.shape[0]==1:
    	X_fmt = str(list(X[0]))
    else:
        X_fmt = str(list(X))
    print(len(X[0]))
    feat_names = str(list(df.drop(columns = ["SK_ID_CURR","TARGET"]).columns))
    return str(probability) + "#" + X_fmt + "#" + feat_names

@app.route('/shap/<client>', methods = ["POST"])
def score_interpretation(client):
    f=open('static/first_shap_values.json')
    shap_values_ = json.load(f)
    try:
        shapley = shap_values_[str(client)][0]
    except:
        shapley = shap_values_['332040'][0]    
    return str(shapley)

@app.route('/expected_value', methods = ["POST"])
def expected_value():
    X_train_neg = np.loadtxt("static/X_train_neg.csv", delimiter=",")
    # y_train_neg = np.loadtxt("static/y_train_neg.csv", delimiter=",")
    # X_test = np.loadtxt("static/X_test.csv", delimiter=",")
    model = joblib.load('static/best_model.pkl')
    train_data = X_train_neg[:,1:]
    train_sample_size = 200
    train_data_sampled = shap.sample(train_data, train_sample_size, random_state=0)
    explainer = shap.KernelExplainer(model.predict_proba, train_data_sampled)
    expected_value = explainer.expected_value[0]
    return str(expected_value)
    

if __name__ == "__main__":
    app.run(debug=True, host='127.0.0.1', port=5001)
