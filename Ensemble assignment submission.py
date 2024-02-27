######################################### Ensemble Techniques Assignment Submission ##############################################
"""

Please ensure you update all the details:
    
Name:        CH Pavan kumar

Batch Id:    15 Sep 2023

Topic:       Ensemble Techniques Assignment

"""

''
'''
CRISP-ML(Q):

Business Understanding: 

Business Problem:       Understanding factors contributing to high sales in a cloth manufacturing company.
Business Objective:    Maximize sales revenue and profit margins through effective identification of key contributing attributes.
Business Constraint:   Minimize resource utilization and time investment in model development and implementation.

Success Criteria:

Business:          Increase sales by at least 10% within the next fiscal year.
Machine Learning:  Achieve a model accuracy of at least 80% in predicting sales.
Economic:         Generate a return on investment (ROI) of 15% from the implementation of optimized strategies based on model insights.


data collection & Description :
    
    Sales: The sales amount, presumably in a certain currency (float64).
    
    CompPrice: The price of the product in comparison with competitors (int64).
    
    Income: The income level of customers (int64).
    
    Advertising: The amount spent on advertising the product (int64).
    
    Population: The population of the area where sales were made (int64).
    
    Price: The price of the product (int64).
    
    ShelveLoc: The location of the product on the shelves (object).
    
    Age: The age of the customers (int64).
    
    Education: The education level of customers (int64).
    
    Urban: Whether the sales were made in an urban area or not (object).
    
    US: Whether the sales were made in the US or not (object).

'''
  
# Code Modularity should be maintained
# Import all required libraries

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sqlalchemy import create_engine
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from feature_engine.outliers import Winsorizer

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV , cross_validate
from sklearn.ensemble import BaggingClassifier, RandomForestClassifier
from sklearn import tree
import sklearn.metrics as skmit

import joblib
import pickle

# Load the data
data = pd.read_csv(r'/Users/pavankumar/Documents/My Learning /Data Sciences/Assignment Question files/Ensemble /ClothCompany_Data (3).csv')


# Connecting to the sql 
user = 'root'
pw = '98486816'
db = 'Clustering'
engine = create_engine(f'mysql+pymysql://{user}:{pw}@localhost/{db}')

data.to_sql('Sales1', con = engine, if_exists = 'replace', chunksize = 1000, index = False)

# retrive the data info()
data.info()
data.head()

# Check the classes in categorical features and also balance
categorical_features = data.select_dtypes(include = ['object']).columns

for i in categorical_features:
    print("\n\n","The Class :",data[i].unique(),"\n")
    print(data[i].value_counts(),"\n")
    print(data[i].value_counts()/len(data[i]),"# In Percentage")
    
# check for null values 
data.isnull().sum()     # No null values

'''
# Auto EDA 
import dtale
d = dtale.show(data)
d.open_browser()

import sweetviz

sv = sweetviz.analyze(data)
sv.show_html("The_report.html")
'''
###################################

# Extrat the features and target variable from the data.
predictors = data.iloc[:,1:]
target = data.iloc[:,:1]

target.rename(columns = {'Sales' :'Sales1' }, inplace = True)

target.describe()

# Discritization of the numeric values into  3 categorical
#target['Sales1'] = pd.cut(target['Sales1'],bins = 3, include_lowest = True, labels = ['Low','Medium','High']) 
# Discritization of the numeric values into  2 categorical
target['Sales1'] = pd.cut(data.Sales,
                              bins = [min(data.Sales),data.Sales.mean(),max(data.Sales)],
                              include_lowest = True,
                              labels = ['Low', 'High'])
con = pd.concat([target,data.iloc[:,:1]], axis = 1)

'''
con['Sales2'] = pd.cut(data.Sales,
                              bins = [min(data.Sales),data.Sales.mean(),max(data.Sales)],
                              include_lowest = True,
                              labels = ['Low', 'High'])'''

# Segregate the numeric and non numeric columns in predictors

numerical_features = predictors.select_dtypes(exclude = ['object']).columns
numerical_features

categorical_features = predictors.select_dtypes(include = ['object']).columns
categorical_features

# Define pipeline for Data Pre-processing

num_pipeline = Pipeline([('impute',SimpleImputer(strategy = 'mean')),('Scale', MinMaxScaler())])

categ_pipeline = Pipeline([('Onehot',OneHotEncoder(drop = 'first'))])

# Process pipeline 
process_pipeline = ColumnTransformer([('numeric',num_pipeline,numerical_features),
                                      ('categorical',categ_pipeline,categorical_features)],
                                     remainder = 'passthrough')

processed = process_pipeline.fit(predictors)

# save the model
joblib.dump(processed,'PData-rocessed-1')

# data clean
data_clean = pd.DataFrame(processed.transform(predictors), columns = processed.get_feature_names_out())
data_clean.info()

# Outlier analysis
columns = list(data_clean.iloc[:,:7].columns)

# Box plot outlier detection
data_clean[columns].plot(kind = 'box', subplots = True, sharey = False, figsize = (12,8))
plt.subplots_adjust(wspace = 0.75)
plt.show()

# We identify few outliers  so we go with winsorizer

winsor = Winsorizer(capping_method = 'iqr',
                    tail = 'both',
                    fold = 1.5,
                    variables = columns)

outliers = winsor.fit(data_clean[columns])

# save the model
joblib.dump(outliers,'winsor')

data_clean[columns] = outliers.transform(data_clean[columns])

# After winsorizer
data_clean[columns].plot(kind = 'box', subplots = True, sharey = False, figsize = (12,8))
plt.subplots_adjust(wspace = 0.75)
plt.show()

# END Data Preprocessing ########################################################################
# Supervised Learning

# split the data into train and test samples using stratified technique
x_train,x_test,y_train,y_test = train_test_split(data_clean, target, test_size = 0.2, stratify = target, random_state = 0)

x_train
x_test

# Model building
# Bagging

clf = tree.DecisionTreeClassifier()

bag = BaggingClassifier(estimator = clf, n_estimators = 500, bootstrap = True, n_jobs = -1, random_state = 42)

# Train the model
bag_clf = bag.fit(x_train,y_train)    

# Predict
train_pred = bag_clf.predict(x_train)

# Train score and confusion matrix
print(f"Train Score for Baaging : {bag_clf.score(x_train,y_train):.3f}")
print(f"Train Confusion matrix : {skmit.confusion_matrix(train_pred,y_train)}")

# Evaluate on the test data
#Predict
pred = bag_clf.predict(x_test)

# Test score and confusion matrix
test_score = bag_clf.score(x_test,y_test)

print(f"Test Score for Bagging : {bag_clf.score(x_test,y_test):.3f}")
print(f"Test Confusion matrix : {skmit.confusion_matrix(pred,y_test)}")

# Confusion Matrix
cm = skmit.confusion_matrix(y_test, pred)

cmplot = skmit.ConfusionMatrixDisplay(cm, display_labels = ['Low','High'])
cmplot.plot()
cmplot.ax_.set(title = 'Bagging model Sales predictions using confusion matrix',
               xlabel = 'Predictions', ylabel = 'Actual')

# save the model 
pickle.dump(bag_clf,open('bag_clf.pkl','wb'))

# Cross validation 
def cross_validation(model, x, y, cv = 5):
    _scoring = ['accuracy','precision','recall','f1']
    results = cross_validate(model, X = x, y=y, cv = cv,
                             scoring = _scoring,
                             return_train_score = True)
    return pd.DataFrame({"Training Accuracy scores": results['train_accuracy'],
          "Mean Training Accuracy": results['train_accuracy'].mean()*100,
          "Training Precision scores": results['train_precision'],
          "Mean Training Precision": results['train_precision'].mean(),
          "Training Recall scores": results['train_recall'],
          "Mean Training Recall": results['train_recall'].mean(),
          "Training F1 scores": results['train_f1'],
          "Mean Training F1 Score": results['train_f1'].mean(),
          "Validation Accuracy scores": results['test_accuracy'],
          "Mean Validation Accuracy": results['test_accuracy'].mean()*100,
          "Validation Precision scores": results['test_precision'],
          "Mean Validation Precision": results['test_precision'].mean(),
          "Validation Recall scores": results['test_recall'],
          "Mean Validation Recall": results['test_recall'].mean(),
          "Validation F1 scores": results['test_f1'],
          "Mean Validation F1 Score": results['test_f1'].mean()
          })

# Visualize
def plot_results(x_label, y_label, title, train_data, val_data):
    
    plt.figure(figsize = (12,8))
    labels = ['1st Fold','2nd Fold','3rd Fold','4th Fold','5th FOld']
    x_axis = np.arange(len(labels))
    plt.gca()
    plt.ylim(0.4, 1)
    plt.bar(x_axis - 0.2, train_data, 0.4, color = 'blue', label = 'Training')
    plt.bar(x_axis + 0.2, val_data, 0.4, color = 'red', label = 'Validation')
    plt.title(title, fontsize = 30)
    plt.xlabel(x_label, fontsize = 14)
    plt.ylabel(y_label, fontsize = 14)
    plt.legend()
    plt.grid()
    plt.show()

# Cross Validation Scores & graph
bagging_scores = cross_validation(bag_clf,x_train,y_train,cv = 5)

# Plot
model_name = 'Bagging Model'
plot_results("Accuracy in 5 Folds", 
             "Accuracy", 
             model_name,
             bagging_scores["Training Accuracy scores"], 
             bagging_scores["Validation Accuracy scores"])
#######################################################################

# Random Forest Classifier

rf_model = RandomForestClassifier()

# Hyperparameters
# Number of Tree's
n_estimators = [int(x) for x in np.linspace( start = 20, stop = 600, num = 10)] # 10

# max features 
max_features = ['auto','sqrt']

# maximum number of trees
max_depth = [2,4]

# minnimu number of samples required to split the node
min_samples_split = [2, 5]

# minnimu number of samples required to leaf the node
min_samples_leaf = [1,2]

# Method of selecting samples for training each tree
bootstrap = [True, False]

# Param grid
param_grid = {'n_estimators' : n_estimators,
              'max_features' : max_features,
              'max_depth' : max_depth,
              'min_samples_split' : min_samples_split,
              'min_samples_leaf' : min_samples_leaf,
              'bootstrap' : bootstrap}
  
# Hyper parameter tunninng with GridSearchCV
rf_gscv = GridSearchCV(rf_model, param_grid, n_jobs = -1, cv = 10, verbose = 1)

# Train the grid search model
random_gscv_model = rf_gscv.fit(x_train,y_train)

# Best Params
rf_gscv.best_params_

# Best Estimator
rf_gscv.best_estimator_

# Predict
pred_train_1 = rf_gscv.predict(x_train)

# Train score and Confusion matrix
print(f"Train Score: {skmit.accuracy_score(y_train, pred_train_1):.3f}\n")
print(f"Train Confusion Matrix : {skmit.confusion_matrix(y_train, pred_train_1)}\n")

# Evaluate on test data
pred_g = rf_gscv.predict(x_test)

#  Test Score & Confusion matrix
print(f"Test score :{skmit.accuracy_score(y_test, pred_g):.3f}\n")
print(f"Test Confusion matrix : {skmit.confusion_matrix(y_test, pred_g)}")

# Confusion Matrix
cm = skmit.confusion_matrix(y_test, pred_g)

cmplot = skmit.ConfusionMatrixDisplay(cm, display_labels = ['Low','High'])
cmplot.plot()
cmplot.ax_.set(title = 'Grid Search Sales predictions using confusion matrix',
               xlabel = 'Predictions', ylabel = 'Actual')

#---------------------------------------------------------------------------------------

# RandomizedSearchCV
rf_rscv = RandomizedSearchCV(rf_model, param_distributions = param_grid, n_jobs = -1, cv = 10, verbose = 2)

# Train the random search model
rf_rscv.fit(x_train,y_train)

# Best Params
rf_rscv.best_params_

# Best Estimator
rf_rscv.best_estimator_

# Evaluate on the test data
# Predict
pred_r = rf_rscv.predict(x_test)

print(f"Test score :{skmit.accuracy_score(y_test, pred_r):.3f}\n")
print(f"Test Confusion matrix : {skmit.confusion_matrix(y_test, pred_r)}")

# Save the model 
pickle.dump(rf_rscv,open('Random_rscv.pkl','wb'))

# Confusion Matrix
cm = skmit.confusion_matrix(y_test, pred_r)

cmplot = skmit.ConfusionMatrixDisplay(cm, display_labels = ['Low', 'High'])
cmplot.plot()
cmplot.ax_.set(title = 'Random Search Sales predictions using confusion matrix',
               xlabel = 'Predictions', ylabel = 'Actual')

# Cross Validation
random_rscv_scores = cross_validation(rf_rscv, x_train, y_train, cv = 5)

# plot
model_name = 'Random Forest Model'
plot_results("Accuracy in 5 Folds", 
             "Accuracy", 
             model_name,
             random_rscv_scores["Training Accuracy scores"], 
             random_rscv_scores["Validation Accuracy scores"])
############################################################################################

# Ada Boosting Model
from sklearn.ensemble import AdaBoostClassifier, GradientBoostingClassifier

ada_clf = AdaBoostClassifier(learning_rate = 0.02, n_estimators = 5000)

ada_clf.fit(x_train,y_train)

# Evaluate on train data
print(f"Train Ada Score : {ada_clf.score(x_train,y_train)}\n")
# Evaluate on train data
print(f"Test Ada Score : {ada_clf.score(x_test,y_test)}\n")

# Save the Model
pickle.dump(ada_clf, open('Ada_model.pkl', 'wb'))
##############################################################
# Gradient Boosting Model

gb_clf = GradientBoostingClassifier()

# Train the model
gb_clf.fit(x_train,y_train)

# Evaluate on train data
print(f"Train GB Score : {gb_clf.score(x_train,y_train)}\n")
# Evaluate on train data
print(f"Test GB Score : {gb_clf.score(x_test,y_test)}\n")

# With Parameters
gb_clf1 = GradientBoostingClassifier(n_estimators= 1000, learning_rate = 0.02, max_depth = 1)

# Train the model
gb_clf1.fit(x_train,y_train)

# Evaluate on train data
print(f"Train GB1 Score : {gb_clf1.score(x_train,y_train)}\n")
# Evaluate on train data
print(f"Test GB1 Score : {gb_clf1.score(x_test,y_test)}\n")

# Save the model
pickle.dump(gb_clf1,open("GB1 Model.pkl", 'wb'))

#######################################################################3
# XGBoosting Model
import xgboost
from xgboost import XGBClassifier

xgb_clf = XGBClassifier(max_depth = 5, n_estimators = 10000, learning_rate = 0.3, n_jobs = -1)

# Label Encoded y_train 
lbe = LabelEncoder()
encoded_y = lbe.fit_transform(y_train)
encoded_test_y = lbe.fit_transform(y_test)
# Train the model
xgb_clf.fit(x_train, encoded_y)

## Evaluate on train data
print(f"Train XGB Score : {xgb_clf.score(x_train,encoded_y)}\n")
# Evaluate on train data
print(f"Test XGB Score : {xgb_clf.score(x_test,encoded_test_y)}\n")

xgboost.plot_importance(xgb_clf)

fi = pd.DataFrame(xgb_clf.feature_importances_.reshape(1, -1), columns = x_train.columns)
fi
# Save the model 
pickle.dump(xgb_clf,open('xgb_clf.pkl','wb'))
# With Random search CV 
xgb = XGBClassifier(n_estimators = 500, learnin_rate = 0.1, random_state = 42)
# params grid
params_grid = {"max_depth" : (3,10,2),
               "gamma" : [0.1,0.2,0.3],
               "subsamle" : [0.8,0.9],
               "colsample_bytree": [0.8, 0.9]}

xgb_clf_rcv = RandomizedSearchCV(xgb, param_distributions = params_grid, cv = 5, n_jobs = -1, verbose = 2)

xgb_clf_rcv.fit(x_train, encoded_y)

# Best Params
xgb_clf_rcv.best_params_

# Best Estimator
xgb_clf_rcv.best_estimator_

## Evaluate on train data
print(f"Train XGBR Score : {xgb_clf_rcv.score(x_train,encoded_y)}\n")
# Evaluate on train data
print(f"Test XGBR Score : {xgb_clf_rcv.score(x_test,encoded_test_y)}\n")

# Save the model 
pickle.dump(xgb_clf_rcv,open('xgb_clf_rcv.pkl','wb'))
##########################################################################
# Ensemble Voting (Hard)
from sklearn.ensemble import VotingClassifier

# Combine all five Voting Ensembles
estimators = [('Bagging',bag_clf),('Random Forest',rf_rscv),('Ada',ada_clf),('GB',gb_clf),('XGBR',xgb_clf_rcv)]

# Instantiate the voting classifier
ensemble_H = VotingClassifier(estimators, voting = 'hard')
ensemble_S = VotingClassifier(estimators, voting = 'soft')
# Train the model
hard_voting = ensemble_H.fit(x_train,y_train)
soft_voting = ensemble_S.fit(x_train,y_train)

# Hard Voting
## Evaluate on train data
print(f"Train Ensemble Score : {hard_voting.score(x_train,y_train)}\n")
# Evaluate on train data
print(f"Test Ensemble Score : {hard_voting.score(x_test,y_test)}\n")

# Soft Voting
## Evaluate on train data
print(f"Train Ensemble Score : {soft_voting.score(x_train,y_train)}\n")
# Evaluate on train data
print(f"Test Ensemble Score : {soft_voting.score(x_test,y_test)}\n")

# ALL Test Results
print("\nAll Test Score Results from all models :\n")
print(f"Bagging :{bag_clf.score(x_test, y_test)}\n")
print(f"Random Forest :{rf_rscv.score(x_test, y_test)}\n")
print(f"Ada Boosting :{ada_clf.score(x_test, y_test)}\n")
print(f"GBoosting :{gb_clf.score(x_test, y_test)}\n")
print(f"XGBoosting :{xgb_clf_rcv.score(x_test, encoded_test_y)}\n")
print(f"Hard Ensemble Score : {hard_voting.score(x_test,y_test)}\n")
print(f"Soft Ensemble Score : {soft_voting.score(x_test,y_test)}\n")

# Save the model
pickle.dump(ensemble_H,open('ensemble_H.pkl','wb'))
pickle.dump(ensemble_S,open('ensemble_S.pkl','wb'))


# We see the value from the  Ada Boosting is more compare to all others for both 3 classifictin and two classification models of sales category
# So we go with Ada Boosting Model




