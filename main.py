import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

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


## RandomForest model
model_1 = RandomForestClassifier(random_state=420)
## XGBoost model
model_2 = XGBClassifier(n_estimators=300,learning_rate=0.1,use_label_encoder=False,eval_metric='error')


## Making Pipeline (even though its not necessary here and we could just impute it without one)
transformer_1 = Pipeline(steps=[('imp_age',SimpleImputer())])
transformer_2 = Pipeline(steps=[('imp_emb',SimpleImputer(strategy='most_frequent'))])
preprocessor = ColumnTransformer(transformers=[('trans_age',transformer_1,['Age']),('trans_emb',transformer_2,['Embarked'])])

## define function to run model with cross-validation and print out error
def run_model(model):
    pipe = Pipeline(steps=[('preprocessor',preprocessor),('model',model)])
    error = cross_val_score(pipe,X,y,cv=10,scoring='accuracy')
    print(error.mean())
    
run_model(model_2)

#TODO:
# look into early_stopping_rounds for xgboost
# PCA for ticketclass and fare
# EDA including plots 
# MI scores and correlation
# more model testing
# hyperparameter tuning 
# scaling 

