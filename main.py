import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
import warnings
import time

from xgboost import XGBClassifier

warnings.filterwarnings(action='ignore', category=UserWarning)

## read file into a dataframe and set index
df = pd.read_csv('train.csv')
df=df.set_index('PassengerId')

## select target and features
y=df['Survived']
X=df.drop('Survived',axis=1)

## drop some useless columns or columns with too little information
X=X.drop(['Name','Ticket','Cabin'],axis=1)

## label encoding sex, embarked columns
X['Sex']=X.Sex.apply(lambda x: 1 if x=='male' else 0)
X.Embarked,_ = X.Embarked.factorize()
X.Embarked=X.Embarked.replace(-1,np.nan)

## Making Pipeline (even though its not necessary here and we could just impute it without one)
transformer_1 = Pipeline([('imp_age',SimpleImputer())])
transformer_2 = Pipeline([('imp_emb',SimpleImputer(strategy='most_frequent'))])
preprocessor = ColumnTransformer([('trans_age',transformer_1,['Age']),('trans_emb',transformer_2,['Embarked'])])

## Defining a method to avoid code in the future 
def pipe(model):
    return Pipeline([('preprocessor',preprocessor),('model',model)])

## Hypertuning parameters for Gradient Boosting Tree with GridSearchCV
def hypertune_xgb(model):
    ## Hypertuning n_estimators 
    grid_params_1 = {'model__n_estimators':range(10,100)}
    grid_search_1 = GridSearchCV(estimator=pipe(model),param_grid=grid_params_1,scoring='accuracy',cv=5,n_jobs=4)
    grid_search_1.fit(X,y)
    model.set_params(n_estimators=grid_search_1.best_estimator_.get_params()['model__n_estimators'])
    
    ## Hypertuning max_depth and min_child_weight
    grid_params_2 = {'model__max_depth':range(1,10),'model__min_child_weight':range(1,4)}
    grid_search_2 = GridSearchCV(estimator=pipe(model),param_grid=grid_params_2,scoring='accuracy',cv=5,n_jobs=4)
    grid_search_2.fit(X,y)
    model.set_params(max_depth=grid_search_2.best_estimator_.get_params()['model__max_depth'])
    model.set_params(min_child_weight=grid_search_2.best_estimator_.get_params()['model__min_child_weight'])
    
    ## Hypertuning gamma 
    grid_params_3 = {'model__gamma':[i/100 for i in range(0,11)]}
    grid_search_3 = GridSearchCV(estimator=pipe(model),param_grid=grid_params_3,scoring='accuracy',cv=5,n_jobs=4)
    grid_search_3.fit(X,y)
    model.set_params(gamma=grid_search_3.best_estimator_.get_params()['model__gamma'])
    
    ## Hypertuning subsample and colsample_bytree
    grid_params_4 = {'model__subsample':[i/10 for i in range(5,11)],'model__colsample_bytree':[i/10 for i in range(5,11)]}
    grid_search_4 = GridSearchCV(estimator=pipe(model),param_grid=grid_params_4,scoring='accuracy',cv=5,n_jobs=4)
    grid_search_4.fit(X,y)
    model.set_params(subsample=grid_search_4.best_estimator_.get_params()['model__subsample'])
    model.set_params(colsample_bytree=grid_search_4.best_estimator_.get_params()['model__colsample_bytree'])
    
    ## Hypertuning reg_alpha and reg_lambda
    grid_params_5 = {'model__reg_alpha':[i/100 for i in range (0,10)],'model__reg_lambda':[i/10000 for i in range(0,10)]}
    grid_search_5 = GridSearchCV(estimator=pipe(model),param_grid=grid_params_5,scoring='accuracy',cv=5,n_jobs=4)
    grid_search_5.fit(X,y)
    model.set_params(reg_alpha=grid_search_5.best_estimator_.get_params()['model__reg_alpha'])
    model.set_params(reg_lambda=grid_search_5.best_estimator_.get_params()['model__reg_lambda'])
    
    print('Best Score:\t\t\t\t {}'.format(grid_search_5.best_score_))
    print('Optimal n_estimators:\t {}'.format(model.get_params()['n_estimators']))
    print('Optimal depth:\t\t\t {}'.format(model.get_params()['max_depth']))
    print('Optimal child weight:\t {}'.format(model.get_params()['min_child_weight']))
    print('Optimal gamma:\t\t\t {}'.format(model.get_params()['gamma']))
    print('Optimal subsample:\t\t {}'.format(model.get_params()['subsample']))
    print('Optimal colsample_bytree:{}'.format(model.get_params()['colsample_bytree']))
    print('Optimal reg_alpha:\t\t {}'.format(model.get_params()['reg_alpha']))
    print('Optimal reg_lambda:\t\t {}'.format(model.get_params()['reg_lambda']))

## Method for RandomizedSearchCV  
def random_search(model,grid):
    random_model=RandomizedSearchCV(estimator=pipe(model),param_distributions=grid,scoring='accuracy',
                                    n_jobs=6,n_iter=40,cv=5,random_state=420)
    random_model.fit(X,y)
    print(random_model.best_score_)
    return random_model.best_params_

## Method for GridSearchCV
def grid_search(model,grid):
    grid_model=GridSearchCV(estimator=pipe(model),param_grid=grid,scoring='accuracy',cv=5,n_jobs=6)
    grid_model.fit(X,y)
    print(grid_model.best_score_)
    return grid_model.best_params_

## Model#1: Gradient Boosting Tree with hypertuned parameters
model_1 = XGBClassifier(
    learning_rate =0.05,
    n_estimators=91,
    max_depth=3,
    min_child_weight=1,
    gamma=0,
    subsample=0.9,
    colsample_bytree=0.5,     
    reg_alpha=0.07,
    reg_lambda=0,
    objective= 'binary:logistic',
    scale_pos_weight=1,
    seed=27,
    use_label_encoder=False,
    eval_metric='error')

## Error for Model#1 
error_1=cross_val_score(pipe(model_1),X,y,cv=5,scoring='accuracy').mean()

model_1.fit(X,y)
feature_importance = permutation_importance(model_1, X, y, n_repeats=10,random_state=0)
sns.barplot(x=X.columns,y=feature_importance.importances_mean)

## Model#2: Random Forest with hypertuned parameters
grid={'model__n_estimators':range(30,40),
      'model__max_features':['auto','log2'],
      'model__max_depth':[3,4,5],
      'model__min_samples_leaf':[2,3],
      'model__min_samples_split':[3,4]}
model_2 = RandomForestClassifier(random_state=420,n_jobs=1,n_estimators=35,max_depth=4,
                                 min_samples_leaf=3,min_samples_split=3)
error_2=cross_val_score(pipe(model_2),X,y,cv=5,scoring='accuracy').mean()

#TODO:
# Random Forest / Hypertuning
# PCA for ticketclass and fare
# EDA including plots 
# MI scores and correlation
# more model testing
# scaling 

