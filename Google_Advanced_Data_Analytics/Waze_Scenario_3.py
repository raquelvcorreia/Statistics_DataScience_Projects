

# Part 1: Hypothesis testing
# Part 2: Linear Regression (build a binomial logistic regression model based on multiple variables)
# Part 3: ML model (predict users churn, test 2 tree-based models: random forest and XGBoost)



# Import packages for data manipulation
import numpy as np
import pandas as pd
from scipy import stats
import os
from datetime import datetime

# Packages for visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Pickle allows to save your ML models, to minimise lengthy re-training and allow you to share, commit, and re-load pre-trained machine learning models.
# Once the model has been trained the fit and the write to pickle instructions should be commented out and read one uncommented.
import pickle

# Import packages for data modeling
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, accuracy_score, precision_score, \
recall_score, f1_score, confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, auc, \
confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay, PrecisionRecallDisplay
from sklearn.linear_model import LogisticRegression

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance


## In order to be able to evaluate more columns when using the IDE
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',23)

# Load dataset into dataframe
df = pd.read_csv('waze_dataset.csv')
print(df.head())
print(df.shape)
print(df.info())

# remove ID column since this information will not be used
df = df.drop('ID', axis=1)
# Check for missing values
print('Total amount of missing data: ', df.isna().sum())
print('The column ''label'' is missing 700 values')

# Drop rows with missing data in `label` column
# These represent less than5% of the data and there is not a non-random obvious reason for the missing data
df = df.dropna(subset=['label'])

####################################
#### Part 1: Hypothesis testing ####
####################################

## Is there a statistically significant difference in the mean amount from rides from iPhone users and Android users
# (the column label with missing values is not relevant for this initial analysis).
##  Conduct a two-sample hypothesis test (t-test) to analyze the difference in the mean amount of rides between iPhone users and Android users.

# Create a new column that codes the device type
# iPhone:1 and Android:2

# 1. Create `map_dictionary`
map_dictionary = {'Android': 2, 'iPhone': 1}
# 2. Create new `device_type` column (copy of the df['device'])
df['device_type'] = df['device']
# 3. Map the new column to the dictionary
df['device_type'] = df['device_type'].map(map_dictionary)

# Since were are interested in the relatioship between the device type and the number of drives.
# As a starting point look at the average number of drives fro each device type.
print(df.groupby('device_type')['drives'].mean())

## Hypothesis Testing
# 𝐻0: There is no difference in average number of drives between drivers who use iPhone devices and drivers who use Androids.
# 𝐻𝐴: There is a difference in average number of drives between drivers who use iPhone devices and drivers who use Androids.

# Consider a significance level of 5%, proceed with a two-sample t-test

# 1. Isolate the `drives` column for iPhone users.
iPhone = df[df['device_type'] == 1]['drives']
# 2. Isolate the `drives` column for Android users.
Android = df[df['device_type'] == 2]['drives']
# 3. Perform the t-test. With equal_var = False stats. test_ind() will perform the unequal variances  t-test (known as Welch's t-test).
print(stats.ttest_ind(a=iPhone, b=Android, equal_var=False))

print('p-value larger than the chosen significance level (5%), null-hypothesis is accepted. There is NO statistically significant difference between drivers who use either iPhone or Androids')


####################################
#### Part 2: Linear Regression  ####
####################################

## Predict user churn based on the available data. using linear regression
## Check the class balance if the target variable: label
print(df['label'].value_counts(normalize=True))


print(df.describe())

print('From the describe output several columns seem to have outliers: session, drives, total_Sessions, total_navigations_fav1, total_navigations_fav2, driven_km_drives and durantion_minutes_drives')


### Create features/columns that could be of interest

## create column: km_per_driving_day
# 1. Create `km_per_driving_day` column
df['km_per_driving_day'] = df['driven_km_drives'] / df['driving_days']
# 2. Call `describe()` on the new column
print(df['km_per_driving_day'].describe())
# The max value is inf this is due to having driving_days = 0, division by zero is undefined and Pandas imputes a value of infinity
# 1. Convert infinite values to zero
df.loc[df['km_per_driving_day']==np.inf, 'km_per_driving_day'] = 0
# 2. Confirm that it worked
print(df['km_per_driving_day'].describe())

## create column: professional_driver
# 1 when user has 60 or more drives and drove more than 15 days last month
df['professional_driver'] = np.where((df['drives'] >= 60) & (df['driving_days'] >= 15), 1, 0)
# 1. Check count of professionals and non-professionals
print(df['professional_driver'].value_counts())
# 2. Check in-class churn rate
print(df.groupby(['professional_driver'])['label'].value_counts(normalize=True))

print('The churn for professional drivers is lower than for non-professional divers: 7.6% vs 19.9% respectively')

# create a copy at this point to use the dataset as is at this point on part 3.
df_3 = df.copy()


### Remove Outliers ####
## Visualize the presence of outliers
fig, axes = plt.subplots(1, 4, figsize=(15, 2))
fig.suptitle('Boxplots for outlier detection')
sns.boxplot(ax=axes[0], x=df['drives'])
sns.boxplot(ax=axes[1], x=df['total_navigations_fav1'])
sns.boxplot(ax=axes[2], x=df['driven_km_drives'])
sns.boxplot(ax=axes[3], x=df['duration_minutes_drives'])
plt.show()

# Outliers can be changed to median, mean, 95th percentile, to a cap based on IQR, etc
# Calculate the 95% percentile for each feature and impute it to outliers

# Impute outliers
for column in ['sessions', 'drives', 'total_sessions', 'total_navigations_fav1',
               'total_navigations_fav2', 'driven_km_drives', 'duration_minutes_drives']:
    threshold = df[column].quantile(0.95)
    df.loc[df[column] > threshold, column] = threshold

print(df.describe())

# Encode categorical variable: label
# create a new column with the binary version in order to keep the original data
df['label2'] = np.where(df['label']=='churned', 1, 0)
print(df[['label', 'label2']].tail())



### Determine if assumptions for logistic regression are met
# check for colinearity among features
# Generate a correlation matrix
print(df.corr(method='pearson'))
# Plot correlation heatmap
plt.figure(figsize=(15,10))
sns.heatmap(df.corr(method='pearson'), vmin=-1, vmax=1, annot=True, cmap='coolwarm')
plt.title('Correlation heatmap indicates many low correlated variables',
          fontsize=18)
plt.show()

## there are two pairs of variables that show colinearity
# sessions and drives: 1.0, driving_days and activity_days: 0.95, duration_minutes_drives and driven_km_drives: 0.69

# Assign X variables use for the model and the  target
# For X drop the following columns: 'label', 'label2', 'device', 'sessions', 'driving_days'
# Labels/target columns will be dropped ('label', 'device_type'), and only one column from the high colinearity pairs will be kept.
# Even though duration_minutes_drives and driven_km_drives show a pearson correlation of 0.69 both will be kept. Although these have a strong correlation the duration wont only depend on the distance.
X = df.drop(columns = ['label', 'label2', 'device', 'sessions', 'driving_days'])

# Isolate target variable
y = df['label2']

### Split the data
# due to the target class imbalance use stratify=y to make sure the split does not result in an under or over representation on the minority class
# stratify =y, tells the function that it should use the class ratio found in the y variable (the target).
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, random_state=42)

## if no scaling is performed then when logist regression  is initiated penalty should be set to None.
model = LogisticRegression(penalty=None, max_iter=400)
model.fit(X_train, y_train)
# get the coeficients (these represent the change in the log odds of the target variable for every one unit increase in X)
print(pd.Series(model.coef_[0], index=X.columns))
# Intercept of the model
print(model.intercept_)


# Get the predicted probabilities of the training data
training_probabilities = model.predict_proba(X_train)
print(training_probabilities)


## the log-odds (a.k.a., logit) of the dependent variable with respect to the predictor variable should be linear.
# 1. Copy the `X_train` dataframe and assign to `logit_data`
logit_data = X_train.copy()
# 2. Create a new `logit` column in the `logit_data` df
logit_data['logit'] = [np.log(prob[1] / prob[0]) for prob in training_probabilities]


# Plot regplot of `activity_days` log-odds
sns.regplot(x='activity_days', y='logit', data=logit_data, scatter_kws={'s': 2, 'alpha': 0.5})
plt.title('Log-odds: activity_days')
plt.show()


#### Results and Evaluation of the Linear regression model ###
# Generate predictions on X_test
y_preds = model.predict(X_test)
# Score the model (accuracy) on the test data
print(model.score(X_test, y_test))

# Show the results with a confusion matrix
cm = confusion_matrix(y_test, y_preds)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                              display_labels=['retained', 'churned'],
                              )
disp.plot()
plt.show()


# Create a classification report
target_labels = ['retained', 'churned']
print(classification_report(y_test, y_preds, target_names=target_labels))


## Graphical representation of the model features
# Create a list of (column_name, coefficient) tuples
feature_importance = list(zip(X_train.columns, model.coef_[0]))

# Sort the list by coefficient value
feature_importance = sorted(feature_importance, key=lambda x: x[1], reverse=True)
print(feature_importance)
# Plot the feature importances

sns.barplot(x=[x[1] for x in feature_importance],
            y=[x[0] for x in feature_importance],
            orient='h')
plt.title('Feature importance')
plt.show()

####################################
####       Part 3: ML model     ####
####################################

# Tree-based modeling techniques to predict on a binary target class.
# The goal of this model is to predict whether or not a Waze user is retained or churned (find factors that drive user churn).

## Selecting a metric
# Implications of false negatives: Waze will fail to implement/deploy measures to retain users that are about to stop using the app
# Implication of false positives: Waze take proactively measures to retain users that can lead to annoyance or negative experiences for lower users

# the risks of false negatives seem more detrimental. However, followup analysis intended to measure the possible negative effects of false positives should be performed.

# Feature Engineering, starting with df_3 generated at the beginning of feature engineering on Part 2.

print(df_3.describe())

# Create a column/feature `percent_sessions_in_last_month`
df_3['percent_sessions_in_last_month'] = df_3['sessions'] / df_3['total_sessions']

# Create a column/feature `total_sessions_per_day`
df_3['total_sessions_per_day'] = df_3['total_sessions'] / df_3['n_days_after_onboarding']

# Create a column/feature `km_per_hour`
df_3['km_per_hour'] = df_3['driven_km_drives']/df_3['duration_minutes_drives']/60

# Create a column/feature `km_per_drive`
df_3['km_per_drive'] = df_3['driven_km_drives']/df_3['drives']
# similar to the feature 'km_per_driving_day' this one also has inf due to the presence of 0 in the column driven_km_drives and these will also have to be imputed to 0
df_3.loc[df_3['km_per_drive']==np.inf, 'km_per_drive'] = 0

# Create a column/feature `percent_of_sessions_to_favorite`
df_3['percent_of_drives_to_favorite'] = (
    df_3['total_navigations_fav1'] + df_3['total_navigations_fav2']) / df_3['total_sessions']

## since tree-based models are robust to outliers and these are the type of models that will be explored the outliers will not be addressed

print(df_3.info())

# convert the target variable (categorical) into a binary column.
# There is only one other categorical column device, however device_type has the same information but in the binary form. Drop device column
df_3['label2'] = np.where(df_3['label']=='churned', 1, 0)
df_3 = df_3.drop(columns = ['device', 'label'])
#check how balanced the dataset is
print(df_3['label2'].value_counts(normalize=True))

# ~18% of the users in the dataset churned. The dataset is unbalanced, however not to an extreme. No rebalancing will be performed however spliting will be stratified

# which metric to use? do to the dataset imbalance accuracy is not the best metric, the model could have high accuracy and still fail to predict the minority class
# As described earlier, false negatives seem more detrimental, thus the model will be selected based on the recall

# data split to a final ratio of 60/20/20 for training/validation/test sets

X = df_3.drop(['label2'], axis=1)
y= df_3['label2']
X_tr, X_test, y_tr, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state = 42)
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, stratify=y_tr, test_size=0.25, random_state = 42)
# check the shapes and partitioning
print(X_train.shape, X_val.shape, X_test.shape, y_train.shape, y_val.shape, y_test.shape)
for x in [X_train, X_val, X_test]:
    print(len(x))



# Define a path to the folder where models will be saved
path = os.getcwd()+'\\'

def write_pickle(path, model_object, save_name:str):
    '''
    save_name is a string.
    '''
    with open(path + save_name + '.pickle', 'wb') as to_write:
        pickle.dump(model_object, to_write)

def read_pickle(path, saved_model_name:str):
    '''
    saved_model_name is a string.
    '''
    with open(path + saved_model_name + '.pickle', 'rb') as to_read:
        model = pickle.load(to_read)

        return model




## Build a random forest model
rf = RandomForestClassifier(random_state=42)
cv_params = {'max_depth': [5, 7, None],
             'max_features': [0.3, 0.6, 1.0],
            #  'max_features': 'auto'
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [2],
             'min_samples_split': [2],
             'n_estimators': [75,150,300],
             }

scoring = ['accuracy', 'precision', 'recall', 'f1']
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='recall')

start = datetime.now()
rf_cv.fit(X_train, y_train)
print(datetime.now() - start)
write_pickle(path, rf_cv, 'rf_waze')

# rf_cv = read_pickle(path, 'rf_waze')
print('The best recall obtained from the GridSearch was ', rf_cv.best_score_)
print(rf_cv.best_params_)


## use the function make_results to output all the scores from the model
def make_results(model_name:str, model_object, metric:str):
    '''
    Arguments:
    model_name (string): a clear/informative name for the table
    model_object: a fit GridSearchCV object
    metric (string): precision, recall, f1, or accuracy

    Returns a pandas df with the F1, recall, precision, and accuracy scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'precision': 'mean_test_precision',
                 'recall': 'mean_test_recall',
                 'f1': 'mean_test_f1',
                 'accuracy': 'mean_test_accuracy',
                 }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame({'model': [model_name],
                        'precision': [precision],
                        'recall': [recall],
                        'F1': [f1],
                        'accuracy': [accuracy],
                        },
                       )

    return table

results = make_results('RF cv', rf_cv, 'recall')
print(results)

## Although the accuracy was not bad the recall is very low for this model.

### Build a XGBoost model

# Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [4,8,12],
             'min_child_weight': [3, 5],
             'learning_rate': [0.01, 0.1],
             'n_estimators': [300, 500]
             }

# Define a dictionary of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']


# Instantiate the GridSearchCV object
xgb_cv = GridSearchCV(xgb, cv_params, scoring=scoring, cv=5, refit='recall')

start = datetime.now()
xgb_cv.fit(X_train, y_train)
print(datetime.now() - start)
write_pickle(path, xgb_cv, 'xgb_waze')

#xgb_cv = read_pickle(path, 'xgb_waze')

print('The best recall obtained from the GridSearch (XGBoost) was ', xgb_cv.best_score_)
print(xgb_cv.best_params_)


xgb_cv_results = make_results ('XGBoost CV', xgb_cv, 'recall')
results = pd.concat([results, xgb_cv_results], axis=0)
print(results)


# XGBoost metrics are also very low. However, there is some improvement relative to the RF model on both recall and F1-scores.


## Model selection
#check the models on the validation set
rf_val_preds = rf_cv.best_estimator_.predict(X_val)
xgb_val_preds = xgb_cv.best_estimator_.predict(X_val)




def get_test_scores(model_name:str, preds, y_test_data):
    '''
    Generate a table of test scores.

    In:
        model_name (string): model's name in the output table
        preds: numpy array of test predictions
        y_test_data: numpy array of y_test data

    Out:
        table: a pandas df of precision, recall, f1, and accuracy scores for your model
    '''
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy]
                          })

    return table


rf_val_scores = get_test_scores('RF val', rf_val_preds, y_val)
results = pd.concat([results, rf_val_scores], axis=0)

xgb_val_scores = get_test_scores('XGBoost val', xgb_val_preds, y_val)
results = pd.concat([results, xgb_val_scores], axis=0)
print(results)

# the scores on the validation set were low and similar to the training one. The XGBoost were better and this will be selected (this is still a less than optimal model)
# To evaluate how this model would perform on future unseen data we will use the XGBoost mdoel on the test data.
xgb_test_preds = xgb_cv.best_estimator_.predict(X_test)
xgb_test_scores = get_test_scores('XGB test', xgb_test_preds, y_test)
results = pd.concat([results, xgb_test_scores], axis=0)

print(results)
# the recall is identical for the test and validation sets.
# Although these are not good scores they are in line with what would be expected in terms of performance given the train and validation scores.

# visualize through a confusion matrix
cm = confusion_matrix(y_test, xgb_test_preds, labels=xgb_cv.classes_)
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=['retained', 'churned'])
disp.plot()
plt.show()

# The model predicted a lot more false negatives than false positives.

# Check features importance
plot_importance(xgb_cv.best_estimator_)
plt.show()

## The XGBoost made more use of many of the deatures than did the logistic regrression model (which dependended very heavily in one single feature for the predictions - activity days)
## 6 out of the 10 top features were engineered features, undersocring how that step can boost model performance.

## Overall this model is still not very predictive. The data avalailable (used) is not predictive of the selected target.

### Additional --> optimal decision threshold
# The default decision threshold for most implementations of classification algorithms—including scikit-learn's—is 0.5.
# This means that, in the case of the Waze models, if they predicted that a given user had a 50% probability or greater of churning,
# then that user was assigned a predicted value of 1—the user was predicted to churn. This thershold might not be ideal specially when there is class imbalance.

# Calculate the precision-recall curve for the XGBoost model
display = PrecisionRecallDisplay.from_estimator(
    xgb_cv.best_estimator_, X_test, y_test, name='XGBoost'
    )
plt.title('Precision-recall curve, XGBoost model')
plt.show()

# As recall increases precision decreases, in this scenario that is not a problem. What is the impact of reducing the decision threshold?
# use .predict_proba and then calculate the label considering a 0.4 threshold.
predicted_probabilities = xgb_cv.best_estimator_.predict_proba(X_test)
probs = [x[1] for x in predicted_probabilities]
new_preds = np.array([1 if x >= 0.4 else 0 for x in probs])

print(get_test_scores('XGB, threshold = 0.4', new_preds, y_test))


# recall is better but as expected precision and accuracy go down as a consequence.
# If for a given scenario/task it would be ok to get a precision of ~30% if that would mean a recall score of 0.5 that can be done.
# In this case it would mean 50% of the people that would churn would be identified at the cost of being right when saying someone will churn only 30% of the time.
# Function to find the optimal threshold to obtain a specific recall score.

def threshold_finder(y_test_data, probabilities, desired_recall):
    '''
    Find the threshold that most closely yields a desired recall score.

    Inputs:
        y_test_data: Array of true y values
        probabilities: The results of the `predict_proba()` model method
        desired_recall: The recall that you want the model to have

    Outputs:
        threshold: The threshold that most closely yields the desired recall
        recall: The exact recall score associated with `threshold`
    '''
    probs = [x[1] for x in probabilities]  # Isolate second column of `probabilities`
    thresholds = np.arange(0, 1, 0.001)    # Set a grid of 1,000 thresholds to test

    scores = []
    for threshold in thresholds:
        # Create a new array of {0, 1} predictions based on new threshold
        preds = np.array([1 if x >= threshold else 0 for x in probs])
        # Calculate recall score for that threshold
        recall = recall_score(y_test_data, preds)
        # Append the threshold and its corresponding recall score as a tuple to `scores`
        scores.append((threshold, recall))

    distances = []
    for idx, score in enumerate(scores):
        # Calculate how close each actual score is to the desired score
        distance = abs(score[1] - desired_recall)
        # Append the (index#, distance) tuple to `distances`
        distances.append((idx, distance))

    # Sort `distances` by the second value in each of its tuples (least to greatest)
    sorted_distances = sorted(distances, key=lambda x: x[1], reverse=False)
    # Identify the tuple with the actual recall closest to desired recall
    best = sorted_distances[0]
    # Isolate the index of the threshold with the closest recall score
    best_idx = best[0]
    # Retrieve the threshold and actual recall score closest to desired recall
    threshold, recall = scores[best_idx]

    return threshold, recall


print('Thershold, recall_score', threshold_finder(y_test, predicted_probabilities, 0.5))

## confirm the results from thresold_finder function

probs_2 = [x[1] for x in predicted_probabilities]
new_preds_2 = np.array([1 if x >= 0.157 else 0 for x in probs])

print(get_test_scores('XGB, threshold = 0.5', new_preds_2, y_test))
