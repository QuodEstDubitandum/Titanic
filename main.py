import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
import warnings

from xgboost import XGBClassifier

# warnings.filterwarnings(action='ignore', category=UserWarning)

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

## Model#1: Gradient Boosting Tree 
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
    nthread=4,
    scale_pos_weight=1,
    seed=27,
    use_label_encoder=False,
    eval_metric='error')

## Error for Model#1 and hypertuned n_estimators
error_1=cross_val_score(pipe(model_1),X,y,cv=5,scoring='accuracy').mean()
print(error_1)

## Hypertuning parameters for Gradient Boosting Tree with GridSearchCV
def hypertune_xgb():
    ## Hypertuning n_estimators 
    grid_params_1 = {'model__n_estimators':range(10,100)}
    grid_search_1 = GridSearchCV(estimator=pipe(model_1),param_grid=grid_params_1,scoring='accuracy',cv=5)
    grid_search_1.fit(X,y)
    model_1.set_params(n_estimators=grid_search_1.best_estimator_.get_params()['model__n_estimators'])
    
    ## Hypertuning max_depth and min_child_weight
    grid_params_2 = {'model__max_depth':range(1,10),'model__min_child_weight':range(1,4)}
    grid_search_2 = GridSearchCV(estimator=pipe(model_1),param_grid=grid_params_2,scoring='accuracy',cv=5)
    grid_search_2.fit(X,y)
    model_1.set_params(max_depth=grid_search_2.best_estimator_.get_params()['model__max_depth'])
    model_1.set_params(min_child_weight=grid_search_2.best_estimator_.get_params()['model__min_child_weight'])
    
    ## Hypertuning gamma 
    grid_params_3 = {'model__gamma':[i/100 for i in range(0,11)]}
    grid_search_3 = GridSearchCV(estimator=pipe(model_1),param_grid=grid_params_3,scoring='accuracy',cv=5)
    grid_search_3.fit(X,y)
    model_1.set_params(gamma=grid_search_3.best_estimator_.get_params()['model__gamma'])
    
    ## Hypertuning subsample and colsample_bytree
    grid_params_4 = {'model__subsample':[i/10 for i in range(5,11)],'model__colsample_bytree':[i/10 for i in range(5,11)]}
    grid_search_4 = GridSearchCV(estimator=pipe(model_1),param_grid=grid_params_4,scoring='accuracy',cv=5)
    grid_search_4.fit(X,y)
    model_1.set_params(subsample=grid_search_4.best_estimator_.get_params()['model__subsample'])
    model_1.set_params(colsample_bytree=grid_search_4.best_estimator_.get_params()['model__colsample_bytree'])
    
    ## Hypertuning reg_alpha and reg_lambda
    grid_params_5 = {'model__reg_alpha':[i/100 for i in range (0,10)],'model__reg_lambda':[i/10000 for i in range(0,10)]}
    grid_search_5 = GridSearchCV(estimator=pipe(model_1),param_grid=grid_params_5,scoring='accuracy',cv=5)
    grid_search_5.fit(X,y)
    model_1.set_params(reg_alpha=grid_search_5.best_estimator_.get_params()['model__reg_alpha'])
    model_1.set_params(reg_lambda=grid_search_5.best_estimator_.get_params()['model__reg_lambda'])
    
    print('Best Score:\t\t\t\t {}'.format(grid_search_5.best_score_))
    print('Optimal n_estimators:\t {}'.format(model_1.get_params()['n_estimators']))
    print('Optimal depth:\t\t\t {}'.format(model_1.get_params()['max_depth']))
    print('Optimal child weight:\t {}'.format(model_1.get_params()['min_child_weight']))
    print('Optimal gamma:\t\t\t {}'.format(model_1.get_params()['gamma']))
    print('Optimal subsample:\t\t {}'.format(model_1.get_params()['subsample']))
    print('Optimal colsample_bytree:{}'.format(model_1.get_params()['colsample_bytree']))
    print('Optimal reg_alpha:\t\t {}'.format(model_1.get_params()['reg_alpha']))
    print('Optimal reg_lambda:\t\t {}'.format(model_1.get_params()['reg_lambda']))


# feat_imp = pd.Series(xgb1.booster().get_fscore())

#TODO:
# Random Forest / Hypertuning
# PCA for ticketclass and fare
# EDA including plots 
# MI scores and correlation
# more model testing
# scaling 

