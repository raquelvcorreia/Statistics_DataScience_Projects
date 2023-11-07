# Objective: analize the data collected by the HR department and to build a model that predicts whether an employee will leave the company.
# If it is possible to predict the employees that are likely to quit it may be possible to identify the factors that lead to that.

## Import the necessary packages
# Import packages for data manipulation
import numpy as np
import pandas as pd
import os
from ydata_profiling import ProfileReport
from scipy import stats
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
from sklearn.tree import DecisionTreeClassifier, plot_tree

## In order to be able to evaluate more columns when using the IDE
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',23)


## Load the data
df = pd.read_csv('HR_capstone_dataset.csv')

print(df.head())
print(df.info())
print(df.describe())


# rename columns so that all columns have a standard format, no typos, and are as concise (still informative) as possible
df = df.rename(columns={'Work_accident': 'work_accident',
                          'average_montly_hours': 'average_monthly_hours',
                          'time_spend_company': 'tenure',
                          'Department': 'department'})

# Checking for missing values
print('Total number of missing values: ', df.isna().sum())

# Checking for duplicates
print('Total number of apparent duplicated data: ', df.duplicated().sum())
print('Percentage of duplicates ', ((df.duplicated().sum())/df.shape[0]))

# Inspect some of the duplicates, are this real duplicates?
print(df[df.duplicated()].head())

# It is unlikely that 2 employees would report exactly the same through several continuous variables. Thus, these appear to be real duplicates and can be dropped.


# remove duplicates
df1 = df.drop_duplicates(keep='first')

# Generate a report profile on all the data
profile = ProfileReport(df1, title="Profiling Report HR SM data")
#generate an html report
profile.to_file("Profile_HR_SM_Dataset.html")


# the target variable is the column 'left', is this data balanced?
print(df1['left'].value_counts(normalize = True))



# Identify outliers present in the data. The presence of outliers can already be infered by analysing the profile report and basic descriptive stats.
# However, boxplots allow us to more easily detect outliers.

# distribution of `tenure`
plt.figure(figsize=(6,6))
plt.title('Boxplot to detect outliers for tenure', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=df1['tenure'])
plt.show()


# Outliers are present. Determine the number of rows containing outliers.
# Compute the 25th percentile value in `tenure`
percentile25 = df1['tenure'].quantile(0.25)
# Compute the 75th percentile value in `tenure`
percentile75 = df1['tenure'].quantile(0.75)
# Compute the interquartile range in `tenure`
iqr = percentile75 - percentile25

# Define the upper limit and lower limit for non-outlier values in `tenure`
upper_limit = percentile75 + 1.5 * iqr
lower_limit = percentile25 - 1.5 * iqr
print("Lower limit:", lower_limit)
print("Upper limit:", upper_limit)

# Identify subset of data containing outliers in `tenure`
outliers = df1[(df1['tenure'] > upper_limit) | (df1['tenure'] < lower_limit)]
# Count how many rows in the data contain outliers in `tenure`
print("Number of rows in the data containing outliers in `tenure`:", len(outliers))


### Visualize variables distribution in the context of the 'left' variable. Is the distribution the same if the employees stayed vs the ones that left.

# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))
# Create boxplot showing `average_monthly_hours` distributions for `number_project`, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='average_monthly_hours', y='number_project', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Monthly hours by number of projects', fontsize='14')
# Create histogram showing distribution of `number_project`, comparing employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['number_project']
tenure_left = df1[df1['left']==1]['number_project']
sns.histplot(data=df1, x='number_project', hue='left', multiple='dodge', shrink=2, ax=ax[1])
ax[1].set_title('Number of projects histogram', fontsize='14')
# Display the plots
plt.show()

# There are some differences among the different groups. It is worth mention that everyone or nearly everyone working in more than 6 projects left (the people in this group worked the most amount of hours)
# optimal amount of projects to work on seem to be between 3-4, these groups have the highest  stay/left ratio.
# Considering a 40h per week schedule, the expected average hours per month should be 166.67. Apart from the group of employees assigned up to 2 projects, all other groups are overworked
# Average monthly hours range from close to 200 to above 250.

print('stay/left outcome for employees working in 7 projects: ', df1[df1['number_project']==7]['left'].value_counts())


# Create scatterplot of `average_monthly_hours` versus `satisfaction_level`, comparing employees who stayed versus those who left
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='satisfaction_level', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');
plt.show()


# There is a considerable number of employees that worked over 240h/w and whose levels of satisfaction are close to zero.
# Another group worked normal hours they left but their statisfaction levels we also low although not as low around 0.4
#  The other major group of employees that left had considerably high satisfaction levels in spite of working between 210 and 280h/w

# It if worth noting that the distributions for the different groups has very strange and specific shapes which could be indicative of
# some data manipulation or use synthetic data

# Evaluate the satisfaction levels in function of tenure
fig, ax = plt.subplots(1, 2, figsize = (22,8))

# Create boxplot showing distributions of `satisfaction_level` by tenure, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='satisfaction_level', y='tenure', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by tenure', fontsize='14')

# Create histogram showing distribution of `tenure`, comparing employees who stayed versus those who left
tenure_stay = df1[df1['left']==0]['tenure']
tenure_left = df1[df1['left']==1]['tenure']
sns.histplot(data=df1, x='tenure', hue='left', multiple='dodge', shrink=5, ax=ax[1])
ax[1].set_title('Tenure histogram', fontsize='14')
plt.show()

# the distributions for the employees who left have a significant amount of outliers. Also show that high levels of satisfaction did not directly associate with retention
# the employees with tenures above 6 did not leave.

# Calculate the mean and median satisfaction for the employees who left and who stayed
print(df1.groupby(['left'])['satisfaction_level'].agg([np.mean, np.median]))

# there is a difference with the ones who stayed showing unsurprisingly higher satisfaction levels


# evaluate the salary levels for different tenures
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))
# Define short-tenured employees
tenure_short = df1[df1['tenure'] < 7]
# Define long-tenured employees
tenure_long = df1[df1['tenure'] > 6]
# Plot short-tenured histogram
sns.histplot(data=tenure_short, x='tenure', hue='salary', discrete=1,
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.5, ax=ax[0])
ax[0].set_title('Salary histogram by tenure: short-tenured people', fontsize='14')
# Plot long-tenured histogram
sns.histplot(data=tenure_long, x='tenure', hue='salary', discrete=1,
             hue_order=['low', 'medium', 'high'], multiple='dodge', shrink=.4, ax=ax[1])
ax[1].set_title('Salary histogram by tenure: long-tenured people', fontsize='14');

# long-term employees did not have disproportionate higher salaries


# satisfaction levels and salary (stayed vs left)
# Set figure and axes
fig, ax = plt.subplots(1, 2, figsize = (22,8))
# Create boxplot showing distributions of `satisfaction_level` by tenure, comparing employees who stayed versus those who left
sns.boxplot(data=df1, x='satisfaction_level', y='salary', hue='left', orient="h", ax=ax[0])
ax[0].invert_yaxis()
ax[0].set_title('Satisfaction by salary', fontsize='14')
# Create histogram showing distribution of `tenure`, comparing employees who stayed versus those who left
salary_stay = df1[df1['left']==0]['salary']
salary_left = df1[df1['left']==1]['salary']
sns.histplot(data=df1, x='salary', hue='left',
            multiple='dodge', shrink=0.4, ax=ax[1])
ax[1].set_title('salary histogram', fontsize='14')
plt.show()

# satisfaction does not seem to have a direct correlation with salaries

# Explore a possible correlation between working long hours and the last evaluation
# Create scatterplot of `average_monthly_hours` versus `last_evaluation`
plt.figure(figsize=(16, 9))
sns.scatterplot(data=df1, x='average_monthly_hours', y='last_evaluation', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', label='166.67 hrs./mo.', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by last evaluation score', fontsize='14');
plt.show()

# Two very different population for the employees who left:
# overworked employees that performed very well
# employees that work just under the 167h/w and that  performed poorly
# working long hours does not directly translate to high evaluations
# has observed on a previous plot most employees work over the 167h/w


# Create plot to examine relationship between `average_monthly_hours` and `promotion_last_5years`
plt.figure(figsize=(16, 3))
sns.scatterplot(data=df1, x='average_monthly_hours', y='promotion_last_5years', hue='left', alpha=0.4)
plt.axvline(x=166.67, color='#ff6361', ls='--')
plt.legend(labels=['166.67 hrs./mo.', 'left', 'stayed'])
plt.title('Monthly hours by promotion last 5 years', fontsize='14')
plt.show()


# Not many employees that were promoted in the last 5 y left
# Most employees working the most amount of hours did not get promoted in the last 5 years
# Almost all the employees working very long hours left

# counts by department and its distribution among stay/left
print('Number of emplyees per department:', df1["department"].value_counts())

plt.figure(figsize=(11,8))
sns.histplot(data=df1, x='department', hue='left', discrete=1,
             hue_order=[0, 1], multiple='dodge', shrink=.5)
plt.xticks(rotation='45')
plt.title('Counts of stayed/left by department', fontsize=14)
plt.show()

# none of the departments has a skewed distribution

# look at potential correlations between the different features. This is also part of the Report Profile generated earlier.
plt.figure(figsize=(16, 9))
heatmap = sns.heatmap(df1.corr(), vmin=-1, vmax=1, annot=True, cmap=sns.color_palette("vlag", as_cmap=True))
heatmap.set_title('Correlation Heatmap', fontdict={'fontsize':14}, pad=12);
plt.show()

# As observed from the different plots, the number of projects, monthly hours and evaluation scores all have some level of correlation.
# However, the maximum correlation observed between two features was 0.33 (average monthly hours and number of projects) and 0.35 (left and satisfaction levels)


## Option 1: Logistic Regression model

# Before proceeding the non-numeric variables will need to be encoded: 'department' and 'salary'
# First copy de df so that we can return to it
df2 = df1.copy()

# Encode the 'salary; column as an ordinal numeric category since although this one is categorical it is also ordinal
df2['salary'] = (
    df2['salary'].astype('category')
    .cat.set_categories(['low', 'medium', 'high'])
    .cat.codes
)

# Dummy encode column 'department'
df2 = pd.get_dummies(df2, drop_first=False)
# check the transformations

print(df2.head())

# Create a heatmap to visualize how correlated variables are
plt.figure(figsize=(8, 6))
sns.heatmap(df2[['satisfaction_level', 'last_evaluation', 'number_project', 'average_monthly_hours', 'tenure']]
            .corr(), annot=True, cmap="crest")
plt.title('Heatmap of the dataset')
plt.show()

# since logistic regression is sensitive to outliers we will remove the ones identified earlier for the variable tenure
df_logreg = df2[(df2['tenure'] >= lower_limit) & (df2['tenure'] <= upper_limit)]


# Isolate the outcome variable
y = df_logreg['left']
# Select the features to be used
X = df_logreg.drop('left', axis=1)
# Split the data into training set and testing set, include stratity=y as only 16.7% of the data corresponds to employees who left the company
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=42)

# Construct a logistic regression model and fit it to the training dataset
log_clf = LogisticRegression(random_state=42, max_iter=500).fit(X_train, y_train)


# Get the predictions for the test set using the fitted LG model
y_pred = log_clf.predict(X_test)


### Confusion matrix
log_cm = confusion_matrix(y_test, y_pred, labels=log_clf.classes_)
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm,
                                  display_labels=log_clf.classes_)
log_disp.plot(values_format='')
plt.show()

target_names = ['Predicted would not leave', 'Predicted would leave']
print(classification_report(y_test, y_pred, target_names=target_names))

## Although the precision and accuracy is not too low. The recall is quite low (0.26) which means the amount of false negatives is quite high which
# is the most important metric for the task. We want to accurately predict the employees that might leave.


## Option 1: Tree-based model
# no need to worry about removing outliners so will use the df before dealing with that:df2

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





y = df2['left']
X = df2.drop('left', axis = 1)
# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)



# Contruct a decision tree model
# Instantiate model
tree = DecisionTreeClassifier(random_state=0)
# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }
# Assign a dictionary of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
# Instantiate GridSearch
tree1 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')


start = datetime.now()
tree1.fit(X_train, y_train)
print(datetime.now() - start)
write_pickle(path, tree1, 'tree1_hr')

# tree1 = read_pickle(path, 'tree1_hr')


print(tree1.best_params_)
print(tree1.best_score_)

print("this is strong roc_auc score")


def make_results(model_name: str, model_object, metric: str):
    '''
    Arguments:
        model_name (string): model name that will be shown in the output table
        model_object: a fit GridSearchCV object
        metric (string): precision, recall, f1, accuracy, or auc

    Returns a pandas df with the F1, recall, precision, accuracy, and auc scores
    for the model with the best mean 'metric' score across all validation folds.
    '''

    # Create dictionary that maps input metric to actual metric name in GridSearchCV
    metric_dict = {'auc': 'mean_test_roc_auc',
                   'precision': 'mean_test_precision',
                   'recall': 'mean_test_recall',
                   'f1': 'mean_test_f1',
                   'accuracy': 'mean_test_accuracy'
                   }

    # Get all the results from the CV and put them in a df
    cv_results = pd.DataFrame(model_object.cv_results_)

    # Isolate the row of the df with the max(metric) score
    best_estimator_results = cv_results.iloc[cv_results[metric_dict[metric]].idxmax(), :]

    # Extract Accuracy, precision, recall, and f1 score from that row
    auc = best_estimator_results.mean_test_roc_auc
    f1 = best_estimator_results.mean_test_f1
    recall = best_estimator_results.mean_test_recall
    precision = best_estimator_results.mean_test_precision
    accuracy = best_estimator_results.mean_test_accuracy

    # Create table of results
    table = pd.DataFrame()
    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'F1': [f1],
                          'accuracy': [accuracy],
                          'auc': [auc]
                          })

    return table


tree1_cv_results = make_results('decision tree cv', tree1, 'auc')
print(tree1_cv_results)

# Decision trees can be vulnerable to overfitting, while random forests avoid overfitting by incorporating multiple trees to make predictions.

# Contruct a random forest model
# Instantiate model
rf = RandomForestClassifier(random_state=0)
# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth': [3,5, None],
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }
# Assign a dictionary of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
# Instantiate GridSearch
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')


start = datetime.now()
rf1.fit(X_train, y_train)
print(datetime.now() - start)
write_pickle(path, rf1, 'rf1_hr')

#rf1 = read_pickle(path, 'rf1_hr')


print(rf1.best_score_)
print(rf1.best_params_)



# Get all CV scores
rf1_cv_results = make_results('random forest cv', rf1, 'auc')
print(tree1_cv_results)
print(rf1_cv_results)


def get_scores(model_name: str, model, X_test_data, y_test_data):
    '''
    Generate a table of test scores.

    In:
        model_name (string):  How the model will be named in the output table
        model:                A fit GridSearchCV object
        X_test_data:          numpy array of X_test data
        y_test_data:          numpy array of y_test data

    Out: pandas df of precision, recall, f1, accuracy, and AUC scores for your model
    '''

    preds = model.best_estimator_.predict(X_test_data)

    auc = roc_auc_score(y_test_data, preds)
    accuracy = accuracy_score(y_test_data, preds)
    precision = precision_score(y_test_data, preds)
    recall = recall_score(y_test_data, preds)
    f1 = f1_score(y_test_data, preds)

    table = pd.DataFrame({'model': [model_name],
                          'precision': [precision],
                          'recall': [recall],
                          'f1': [f1],
                          'accuracy': [accuracy],
                          'AUC': [auc]
                          })

    return table

## Continue with the random forest model

## make predictions on the test data and evaluate the scores
rf1_test_scores = get_scores('random forest1 test', rf1, X_test, y_test)
print(rf1_test_scores)

# Test and validation score are very similar and good. No overfitting was detected.



#### repeat DT and RF
# Exclude the possibility of data leakage. Which data might be problematic?
# 'satisfaction_level' the data for all employees may not be available when the model is deployed
# 'average_monthly_hours' these can be impacted either by the fact that some employees have already
# decided they will quit, or they may have already been identified by management as people to be fired.
# Either scenarios could result in employees working fewer hours.

# Drop `satisfaction_level` and save resulting dataframe in new variable
df3 = df2.drop('satisfaction_level', axis=1)


# Create 'overworked' column, initially a copy of the average monthly hours.
df3['overworked'] = df3['average_monthly_hours']

# Inspect max and min average monthly hours values
print('Max hours:', df3['overworked'].max())
print('Min hours:', df3['overworked'].min())

# 167 is the approx average number of monthly hours for someone whos works 50 weeks per year, 5 days per week, 8 hours per day.
# Lets consider overworked someone that works more thatn 180h per month on average (~9h per day)
# Define `overworked` as working > 180 hrs/week. Replace the value son the df3['overworked'] to a binary column, where employees who work over 180h/w are assigned 1.
df3['overworked'] = (df3['overworked'] > 180).astype(int)


# drop the df3['average_monthly_hours']
df3 = df3.drop('average_monthly_hours', axis=1)

# Isolate again the outcome and features and split the data (now with the new features where potential data leakage was addressed)

y = df3['left']
X = df3.drop('left', axis=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, stratify=y, random_state=0)


## DT v2
# Instantiate model
tree = DecisionTreeClassifier(random_state=0)
# Assign a dictionary of hyperparameters to search over
cv_params = {'max_depth':[4, 6, 8, None],
             'min_samples_leaf': [2, 5, 1],
             'min_samples_split': [2, 4, 6]
             }
# Assign a dictionary of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
# Instantiate GridSearch
tree2 = GridSearchCV(tree, cv_params, scoring=scoring, cv=4, refit='roc_auc')

start = datetime.now()
tree2.fit(X_train, y_train)
print(datetime.now() - start)

write_pickle(path, tree2, 'tree2_hr')

#tree2 = read_pickle(path, 'tree2_hr')


print(tree2.best_params_)
print(tree2.best_score_)

tree2_cv_results = make_results('decision tree2 cv', tree2, 'auc')
print(tree1_cv_results)
print(tree2_cv_results)


# RF v2

rf = RandomForestClassifier(random_state=0)
cv_params = {'max_depth': [3,5, None],
             'max_features': [1.0],
             'max_samples': [0.7, 1.0],
             'min_samples_leaf': [1,2,3],
             'min_samples_split': [2,3,4],
             'n_estimators': [300, 500],
             }
scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
rf2 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='roc_auc')

start = datetime.now()
rf2.fit(X_train, y_train)
print(datetime.now() - start)
write_pickle(path, rf2, 'rf2_hr')

#rf2 = read_pickle(path, 'rf2_hr')

print(rf2.best_params_)
print(rf2.best_score_)

rf2_cv_results = make_results('random forest2 cv', rf2, 'auc')
print(tree2_cv_results)
print(rf2_cv_results)

#Using AUC (the model ranks a random positive example more highly than a random negative example) as the deciding metric RF is the best performing model


# Get predictions on test data using the rf2 model
rf2_test_scores = get_scores('random forest2 test', rf2, X_test, y_test)
print(rf2_test_scores)

# Generate array of values for confusion matrix
preds = rf2.best_estimator_.predict(X_test)
cm = confusion_matrix(y_test, preds, labels=rf2.classes_)

# Plot confusion matrix
disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rf2.classes_)
disp.plot(values_format='')
plt.show()

# The model predicts more false positives than false negatives.


# even if the RF is according to the scores the best model will evaluate splits and feature importance for DT model as well as the RF

#### Exploring the DT model
# Plot the tree
plt.figure(figsize=(85,20))
plot_tree(tree2.best_estimator_, max_depth=6, fontsize=14, feature_names=X.columns,
          class_names={0:'stayed', 1:'left'}, filled=True);
plt.show()

#tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_, columns=X.columns)
tree2_importances = pd.DataFrame(tree2.best_estimator_.feature_importances_,
                                 columns=['gini_importance'],
                                 index=X.columns
                                )
tree2_importances = tree2_importances.sort_values(by='gini_importance', ascending=False)

# Only extract the features with importances > 0
tree2_importances = tree2_importances[tree2_importances['gini_importance'] != 0]
print(tree2_importances)


sns.barplot(data=tree2_importances, x="gini_importance", y=tree2_importances.index, orient='h')
plt.title("Decision Tree: Feature Importances for Employee Leaving", fontsize=12)
plt.ylabel("Feature")
plt.xlabel("Importance")
plt.show()

#### Exploring the RF model

# Get feature importances
feat_impt = rf2.best_estimator_.feature_importances_
# Get indices of top 10 features
ind = np.argpartition(rf2.best_estimator_.feature_importances_, -10)[-10:]
# Get column labels of top 10 features
feat = X.columns[ind]
# Filter `feat_impt` to consist of top 10 feature importances
feat_impt = feat_impt[ind]
y_df = pd.DataFrame({"Feature":feat,"Importance":feat_impt})
y_sort_df = y_df.sort_values("Importance")
fig = plt.figure()
ax1 = fig.add_subplot(111)
y_sort_df.plot(kind='barh',ax=ax1,x="Feature",y="Importance")
ax1.set_title("Random Forest: Feature Importances for Employee Leaving", fontsize=12)
ax1.set_ylabel("Feature")
ax1.set_xlabel("Importance")
plt.show()



###############################
##### Final observations ######
###############################


# The tree-based models outperformed the logistic regression
# There was just a modest increment between DT and RF model.
# Feature importance was identical for both models
