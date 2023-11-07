import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import xgboost as xgb
from xgboost import cv


## Load the data into a dataframe
cancer_df = pd.read_csv('cervical_cancer.csv')
#print(cancer_df.head())
#print the last 20 rows of the dataframe
#print(cancer_df.tail(20))

#print(cancer_df.info())

## get the statistics of the dataframe
cancer_df.describe()

## replace the '?', where we have missing data, with nan(not a number)
cancer_df = cancer_df.replace('?', np.nan)

## Visually check if some columns might have a lot of missing values
plt.figure(figsize = (5,5))
sns.heatmap(cancer_df.isnull(), yticklabels = False)
plt.show()

## Count the # of Nan in different columns
nan_count = cancer_df.isna().sum()

## 2 columns: "STDs Time since first diagnosis" and "STDs Time since last diagnosis" have a lot of missing values. So I will drop these two columns.
cancer_df = cancer_df.drop(columns = ['STDs: Time since first diagnosis', 'STDs: Time since last diagnosis'])


## most of the column types are objects and thus it is not possible to get statistics. Convert object type to numeric type
cancer_df = cancer_df.apply(pd.to_numeric)

## To deal with the remain nan values we will replace them with the mean from that column. There are different approaches for dealing with it like
## using the min, max or median value instead of the mean
cancer_df = cancer_df.fillna(cancer_df.mean())

## check if there is still any nan in the df
nan_count = cancer_df.isna().sum()
#print(nan_count) ## as expected none are present

## check the range of ages of the study participants
age_range = [min(cancer_df['Age']), max(cancer_df['Age'])]


## what are the biopsy results for the oldest person in the study and the youngest
young_biop = cancer_df.loc[cancer_df['Age'] == min(cancer_df['Age']), 'Biopsy']
old_biop = cancer_df.loc[cancer_df['Age'] == max(cancer_df['Age']), 'Biopsy']

## Get the correlation matrix
corr_matrix = cancer_df.corr()
plt.figure(figsize = (10,10))
sns.heatmap(corr_matrix, annot = True)
plt.show()

## histogram for the entire DF
cancer_df.hist(bins = 10, figsize = (10,10), color = 'b')
plt.show()



## remove the target, what I would like to predict from the DF. And get the target in a separate DF
input_data = cancer_df.drop(columns = ['Biopsy'])
target_data = cancer_df['Biopsy']


## Look at the correlation of the different features with the Biopsy column
input_data.corrwith(target_data).plot.bar(
    title = 'Correlation With Biopsy',
    figsize = (12, 5),
    cmap = 'plasma')
plt.show()


## convert the df into arrays and the data into 'float32'.
X = np.array(input_data).astype('float32')
y = np.array(target_data).astype('float32')

#scaling the data before feeding the model (different features can be in very different scales)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# define data_dmatrix
data_dmatrix = xgb.DMatrix(data=X,label=y)


## split the data into test and train sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=2)


##  using XGBoost, this algorithm can be used from regression or classification tasks. Here it will be used for classification.
##  It's a supervised learning algorithm and implements gradient boosted trees algorithms. It works by combining the an ensemble of predictions from several weak models
##  Robust to many data distributions and relationships, and it has many hyperparameters to tune the performance
##  Increase speed and memory use

## typical max_depth: 3-10
## Learning rate range = [0,1], typical final values 0.01-0.2

# declare parameters
params = {
            'objective':'binary:logistic',
            'max_depth': 5,
            'alpha': 10,
            'learning_rate': 0.1,
            'n_estimators':100
        }

model = xgb.XGBClassifier(**params)
model.fit(X_train, y_train)

## evaluate how the model is on the train data
result_train = model.score(X_train, y_train)
print('Evaluating performance on the train dataset ', result_train)

## evaluate the model on the test data
result_test = model.score(X_test, y_test)
print('Evaluating performance on the test dataset ', result_test)

## make prediction on the test data using the trained model
y_predict = model.predict(X_test)
print(classification_report(y_test, y_predict))

# compute and print accuracy score
print('XGBoost model accuracy score: {0:0.4f}'. format(accuracy_score(y_test, y_predict)))

## get the confusion matrix
cm = confusion_matrix(y_test, y_predict)
sns.heatmap(cm, annot = True)
plt.show()

## evaluate the effect of the number of estimates and tree depth on the model
model2 = xgb.XGBClassifier(learning_rate = 0.1, max_depth = 50, n_estimators = 200)
model2.fit(X_train, y_train)
result_train_2 = model2.score(X_train, y_train)
result_test_2 = model2.score(X_test, y_test)
print('Evaluating the effect of increasing the max_depth and n_estimators ', result_train_2, result_test_2 )

y_predict_2 = model2.predict(X_test)
print(classification_report(y_test, y_predict_2))
cm_2 = confusion_matrix(y_test, y_predict_2)
sns.heatmap(cm_2, annot = True)
plt.show()



## kfold cross validation using XGBoost
params = {"objective":"binary:logistic",'colsample_bytree': 0.3,'learning_rate': 0.1,
                'max_depth': 5, 'alpha': 10}

xgb_cv = xgb.cv(dtrain=data_dmatrix, params=params, nfold=3,
                    num_boost_round=50, early_stopping_rounds=10, metrics="auc", as_pandas=True, seed=123)

print('Preview the xgb_cv: ', xgb_cv)

## Feature Importance
xgb.plot_importance(model)
plt.figure(figsize = (16, 12))

