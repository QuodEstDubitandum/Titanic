# Estimation of Titanic Survivors using a Kaggle dataset
This repository includes a dataset from Kaggle (train.csv, test.csv) that shows different features of passengers that boarded 
the titanic. The task is to predict which passengers of the test set survived while only knowing whether a passenger of the training set survived or not.
My final RandomForestClassifier model had an overall accuracy of 78.46% on the test set and around 82-83% on the training set while using cross validation
to not overfit on the training set. It seems to be not easy to reach much higher accuracy (highest ones so far that I have seen which do not overfit 
on the test set and do not use the information of the test set to reverse engineer a solution are around 81-82% on the test set) due to the low 
similiarity of the test set and the training set as well as having too little data (891/418 passengers in training/test set).

## Feature Description
- PassengerId: Id of the passenger (integer) <br>
- Pclass: Ticket class of the passenger (either 1,2 or 3) <br>
- Sex: Passengers sex (male or female) <br>
- Age: Age of the passenger (rational number ... why though????) <br>
- SibSp: Number of passengers siblings/spouses (integer) <br>
- Parch: Number of passengers parents/children (integer) <br>
- Ticket: Passengers ticket number (numbers/letters, no real pattern) <br>
- Fare: Passengers ticket price (rational number) <br>
- Cabin: Passengers cabin (numbers/letters, no real pattern) <br>
- Embarked: Port of which the passenger embarked (C,Q or S)

## Deleting features
- Ticket: The ticket number had no real pattern and would have led to noise in the data <br>
- Cabin: The cabin had some pattern which seemed to correlate with Pclass a bit, but the main problem was that 
only 295 of 1309 values were given, meaning that most of the data was missing and imputing all those values would not 
have made any sense

## Imputing
A total of 3 features had to be imputed:
- Fare: The test set was missing 1 value in this feature and i imputed it by the average Fare of the dataset <br>
- Embarked: The training set was missing 3 values which i imputed by the most frequent of the 3 different ports <br>
- Age: This one was a little bit more complex since this feature was missing 263 values
Now it would be kind of inaccurate to impute every single one of these values by the mean or median value, so I looked 
at the correlation of Age and other features so I can use another feature to predict the Age feature.
After seeing that the Pclass and SibSp features had a high negative correlation with Age, I created a feature by adding those 
two features and then used this newly created feature to predict Age.

## Feature Engineering 
Since this dataset provides little information, we will try to get everything out of it that we can.
For this, we do the following things: 
- We replace the passengers Name by the title that is contained in the name (Mr. Mrs. etc.), which could give us information
about the social status of the passenger which could then have direct correlation to survival rate.
- We also split Age into different age groups and replace the rational numbers with categorical variables representing
the age groups
- Lastly we create a new feature called "Family", which is just the addition of the SibSp and Parch feature 
representing the number of family members the passenger has. We also delete SibSp and Parch after that.


## Encoding
- Sex: Binary encoding for male and female (1,0) <br>
- Embarked: One-Hot encoded Embarked <br>
- Family: One-Hot encoded Family <br>
- Age: One-Hot encoded Age <br>
- Name: One-Hot encoded Name 

## Model Selection 
To build a Machine Learning Model from our dataset, I tried 3 different models, which are mostly used on small datasets 
for a high accuracy ML model:
- Random Forest <br>
- Gradient Boosted Tree <br>
- Support Vector Machine <br>

On the test dataset, Random Forest performed the best, while Gradient Boosted Tree performed the best on the training set 
(which does not matter for actual applications though). All of these models were fit to a 5-fold cross validation of the training set
to avoid overfitting.

## Hyperparameter Tuning 
All of these models parameters were then tuned on a 5-fold cross validation of the training set by using Grid Search.
