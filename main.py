import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
import warnings
import time
from xgboost import XGBClassifier

warnings.filterwarnings('ignore')


## Read file into a dataframe and set index
X_train = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

## Select target and features
y=X_train['Survived']
m=y.size
X_train=X_train.drop('Survived',axis=1)

## Combining train and test data to write less code for imputing/encoding etc.
total = pd.concat([X_train,X_test],axis=0)
total = total.set_index('PassengerId')

## Drop some useless columns or columns with too little information
total=total.drop(['Ticket','Cabin'],axis=1)
Corr_Matrix = total.corr()

## Imputing (Embarked, Fare)
## Binary encoding (Sex)
total['Sex']=total.Sex.apply(lambda x: 1 if x=='male' else 0)
total['Embarked']=SimpleImputer(strategy='most_frequent').fit_transform(total[['Embarked']])
total['Fare']=SimpleImputer(strategy='mean').fit_transform(total[['Fare']])

## Stripping titles off of the names of passengers
total['Name']=total['Name'].apply(lambda x: x.split(',')[1].strip().split('.')[0])


## Pclass and SibSp seem to correlate negatively with Age, 
## meaning we can use those features to help impute Age
total['Age_feature']=total['Pclass']+total['SibSp']
quantile_list = total[['Age_feature','Age']].groupby(['Age_feature']).mean()
quantile_list = total['Age_feature'].value_counts().apply(lambda x: x/(total['Age'].size)).sort_index()
quantile_list = quantile_list.tolist()
quantile_list.reverse()
for i in range(1,len(quantile_list)):
    quantile_list[i]=quantile_list[i]+quantile_list[i-1]
age_quantiles = total['Age'].quantile(quantile_list)
total['Age']=total['Age'].where(pd.notnull(total['Age']),other=-1*total['Age_feature'])

## Imputing missing values of Age based on our new feature
def age_imputing(x):
    if x==-1:
        x=50
    elif x==-2:
        x=40
    elif x==-3:
        x=20
    elif x==-4:
        x=10
    else:
        x=5
    return x

## Encoding Age into different quantiles
def age_encoding(x):
    if x<=6:
        x=1
    elif x>6 and x<=18:
        x=2
    elif x>18 and x<=32:
        x=3
    elif x>33 and x<=45:
        x=4
    else:
        x=5
    return x

total['Age']=total['Age'].apply(lambda x: age_imputing(x) if x<0 else x).apply(lambda x: age_encoding(x))

## Creating new feature Family, which holds the number of family members
total['Family']=total['SibSp']+total['Parch']
total=total.drop(['SibSp','Parch','Age_feature'],axis=1)

## One-Hot Encoding our nominal categorical variables
total = pd.get_dummies(total,columns=['Embarked','Age','Family','Name'])

## Dividing into Train and Test split after preprocessing again
X_train = total.iloc[:m] 
X_test = total.iloc[m:]

## Method for RandomizedSearchCV  
def random_search(model,grid,data):
    random_model=RandomizedSearchCV(estimator=model,param_distributions=grid,scoring='accuracy',
                                    n_jobs=6,n_iter=40,cv=5,random_state=420)
    random_model.fit(data,y)
    print(random_model.best_score_)
    return random_model.best_params_


## Method for GridSearchCV
def grid_search(model,grid,data):
    grid_model=GridSearchCV(estimator=model,param_grid=grid,scoring='accuracy',cv=5,n_jobs=6)
    grid_model.fit(data,y)
    print(grid_model.best_score_)
    return grid_model.best_params_



## Model#1: Gradient Boosting Tree 
model_1 = XGBClassifier(
    learning_rate =0.05,
    n_estimators=4920,
    max_depth=4,
    min_child_weight=2,
    gamma=0.5,
    subsample=0.6,
    colsample_bytree=0.6,     
    reg_alpha=1,
    reg_lambda=5,
    objective= 'binary:logistic',
    scale_pos_weight=1,
    seed=27,
    use_label_encoder=False,
    eval_metric='error')

grid_0={'n_estimators':[4915,4920,4925]}
grid_1={'max_depth':range(3,6),
        'min_child_weight':range(1,4)}
grid_2={'gamma':[i/100 for i in range(45,55)]}     
grid_3={'subsample':[0.5,0.6,0.7],
        'colsample_bytree':[0.5,0.6,0.7]}   
grid_4={'reg_alpha':[0.9,1,1.2],
        'reg_lambda':[4,5,6]}

## Errors for CV and whole train set
model_1.fit(X_train,y)
cv_error_xgb=cross_val_score(model_1,X_train,y,cv=5,scoring='accuracy').mean()
model1_pred = model_1.predict(X_train)
train_error_xgb = accuracy_score(y,model1_pred)



## Model#2: Random Forest 
model_2 = RandomForestClassifier(
    random_state=420,
    n_jobs=1,
    n_estimators=100,
    max_depth=5,
    min_samples_leaf=2,
    min_samples_split=2)

rf_grid={'n_estimators':[90,100,110],
         'max_features':['auto','log2'],
         'max_depth':[5,6,7],
         'min_samples_leaf':[2,3,4],
         'min_samples_split':[2,3]}



## Errors for CV and whole train set
model_2.fit(X_train,y)
cv_error_rf=cross_val_score(model_2,X_train,y,cv=5,scoring='accuracy').mean()
model2_pred = model_2.predict(X_train)
train_error_rf = accuracy_score(y,model2_pred)


## Plotting the importance of each feature 
# feature_importance = permutation_importance(model_2, X_train, y, n_repeats=10,random_state=0)
# sns.barplot(x=X_train.columns,y=feature_importance.importances_mean)



## Model#3: Support Vector Machine
## Needs Scaling before fitting 
scaler = StandardScaler()
total_scaled = scaler.fit_transform(total)
X_train_scaled = total_scaled[:m,:] 
X_test_scaled = total_scaled[m:,:]

model_3=SVC(
    random_state=420,
    C=0.5)

svm_grid={'C':[0.4,0.5,0.7],
          'kernel':['rbf','poly'],
          'gamma':['scale','auto']}

## Errors for CV and whole train set
model_3.fit(X_train_scaled,y)
cv_error_svm=cross_val_score(model_3,X_train_scaled,y,cv=5,scoring='accuracy').mean()
model3_pred = model_3.predict(X_train_scaled)
train_error_svm = accuracy_score(y,model3_pred)


## Predicting Test set (currect result on Kaggle: 78.47%)
X_test['Survived'] = model_2.predict(X_test)
X_test = X_test['Survived']

X_test.to_csv('test_results.csv')

