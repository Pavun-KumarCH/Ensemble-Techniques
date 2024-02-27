from flask import  Flask, render_template, request
from sqlalchemy import create_engine

import re
import pandas as pd
import pickle
import joblib

# Data Preprocesseing
data_processed = joblib.load('PData-rocessed-1')
winsor = joblib.load('winsor')
# columns
columns = ['numeric__CompPrice',
           'numeric__Income',
           'numeric__Advertising',
           'numeric__Population',
           'numeric__Price',
           'numeric__Age',
           'numeric__Education']
# Models
bag_model = pickle.load(open('bag_clf.pkl','rb'))
random_rscv = pickle.load(open('random_rscv.pkl','rb'))
ada_model = pickle.load(open("Ada_model.pkl",'rb'))
gb_model = pickle.load(open('GB1 Model.pkl','rb'))
xgb_model = pickle.load(open('xgb_clf_rcv.pkl','rb'))

# Voting models
hard_voting = pickle.load(open('ensemble_H.pkl','rb'))
soft_voting = pickle.load(open('ensemble_S.pkl','rb'))

# Connecting to the sql 
user = 'root'
pw = '98486816'
db = 'Clustering'
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')


app = Flask(__name__)

# Define Flask

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/success', methods = ['POST'])
def success():
    if request.method == 'POST' :
        f = request.files['file']
        data = pd.read_csv(f)
        data_clean = pd.DataFrame(data_processed.transform(data), columns = data_processed.get_feature_names_out())
        data_clean[columns] = winsor.transform(data_clean[columns])
        prediction1 = pd.DataFrame(bag_model.predict(data_clean), columns = ['Bagging Sales'])
        prediction2 = pd.DataFrame(random_rscv.predict(data_clean), columns = ['RFC Sales'])
        prediction3 = pd.DataFrame(ada_model.predict(data_clean), columns = ['ADA Sales'])
        prediction4 = pd.DataFrame(gb_model.predict(data_clean), columns = ['GB Sales'])
        prediction5 = pd.DataFrame(xgb_model.predict(data_clean), columns = ['XGB Sales'])
        
        # Voting model
        voting_h = pd.DataFrame(hard_voting.predict(data_clean), columns = ['Hard Voting Sales'])
        voting_s = pd.DataFrame(soft_voting.predict(data_clean), columns = ['Soft Voting Sales'])
        
        final_data = pd.concat([voting_h, voting_s, prediction1, prediction2, prediction3, prediction4, prediction5, data], axis = 1)
        
        final_data.to_sql('Sales Predictions', con = engine, if_exists = 'replace', chunksize = 1000, index = False)
        
        return render_template("new.html", Y = final_data.to_html(justify='center').replace('<table border="1" class="dataframe">','<table border="1" class="dataframe" bordercolor="#000000" bgcolor="#bdbbb9">'))

if __name__=='__main__':
    app.run(debug = True)

        
        
        
        
        
        
        
        