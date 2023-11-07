## Does a video contain a claim or offers an opinion? Verified vs not_verified users is there a difference?

# Part 1: Hypothesis testing
# Part 2: Linear Regression
# Part 3: ML model for claims classification (does a video contain a claim or offers an opinion?)

# Import packages for data manipulation
import pandas as pd
import numpy as np
import os
from datetime import datetime

# Import packages for data visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Import packages for statistical analysis/hypothesis testing
from scipy import stats

# Pickle allows to save your ML models, to minimise lengthy re-training and allow you to share, commit, and re-load pre-trained machine learning models.
# Once the model has been trained the fit and the write to pickle instructions should be commented out and read one uncommented.
import pickle

# Import packages for data preprocessing
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.utils import resample

# Import packages for data modeling
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report,  accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from xgboost import plot_importance




## In order to be able to evaluate more columns when using the IDE
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',23)

# Load dataset into dataframe
data = pd.read_csv("tiktok_dataset.csv")

print(data.head())
print(data.describe())

# Check for missing values
print('Total amount of missing data: ', data.isna().sum())

# Drop rows with missing values
data = data.dropna(axis=0)

# Check for duplicates
print('Number of duplicated present in the data: ', data.duplicated().sum())

# Get a copy of the dataset to a new dataframe to be used in Part 3
data_3 = data.copy()

# Compute the mean `video_view_count` for each group in `verified_status`
verified_view_count = data.groupby("verified_status")["video_view_count"].mean()
print(verified_view_count)

#################################
### Part1: Hypothesis Testing ###
#################################
#𝐻0: There is no difference in number of views between TikTok videos posted by verified accounts and
# TikTok videos posted by unverified accounts (any observed difference in the sample data is due to chance or sampling variability).

#𝐻𝐴: There is a difference in number of views between TikTok videos posted by verified accounts and
# TikTok videos posted by unverified accounts (any observed difference in the sample data is due to an actual difference in the corresponding population means).

### Proceed with a two-sample t-test and consider a 5% significant level

# Conduct a two-sample t-test to compare means
# Save each sample in a variable: not verified vs verified
not_verified = data[data["verified_status"] == "not verified"]["video_view_count"]
verified = data[data["verified_status"] == "verified"]["video_view_count"]

# Implement a t-test using the two samples
print(stats.ttest_ind(a=not_verified, b=verified, equal_var=False))

### The p-value is extremely small (much smaller than the significance level of 5%), the null hypothesis is rejected.
### There is a statistically significant difference in the mean video view count between verified and unverified accounts on TikTok.

##################################
#### Part2: Linear Regression ####
##################################

### EDA, prepare the data for Linear regression
## Look for data anolamies like outliers and class imbalance that could affect modeling
## Verify model assumptions

## Have a look at how the data looks like and data types for the features/variables
print('Number of entries and features: ', data.shape)
print(data.dtypes)
print(data.info())

# Generate basic descriptive stats
print(data.describe())


### Check for and handle outliers. ###
# Boxplot to visualize distribution of `video_duration_sec`
plt.figure(figsize=(6,2))
plt.title('Boxplot to detect outliers for video_duration_sec', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=data['video_duration_sec'])
plt.show()

# Boxplot to visualize distribution of `video_view_count`
plt.figure(figsize=(6,2))
plt.title('Boxplot to detect outliers for video_view_count', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=data['video_view_count'])
plt.show()

# Boxplot to visualize distribution of `video_like_count`
plt.figure(figsize=(6,2))
plt.title('Boxplot to detect outliers for video_like_count', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=data['video_like_count'])
plt.show()

# Boxplot to visualize distribution of `video_comment_count`
plt.figure(figsize=(6,2))
plt.title('Boxplot to detect outliers for video_comment_count', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
sns.boxplot(x=data['video_comment_count'])
plt.show()


### Video like counts and video comment counts both have some extreme high values.
# To avoid having these dominate the modeling the values for this two features will be capped based on their IQR
## video_like_count capping
percentile25 = data["video_like_count"].quantile(0.25)
percentile75 = data["video_like_count"].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr # use the standard formula
data.loc[data["video_like_count"] > upper_limit, "video_like_count"] = upper_limit

## video_commment_count capping
percentile25 = data["video_comment_count"].quantile(0.25)
percentile75 = data["video_comment_count"].quantile(0.75)
iqr = percentile75 - percentile25
upper_limit = percentile75 + 1.5 * iqr # use the standard formula
data.loc[data["video_comment_count"] > upper_limit, "video_comment_count"] = upper_limit

### Check for class balance. Verified vs not_verified users
# Check class balance
data["verified_status"].value_counts(normalize=True)
print('The two outcome classes are unbalanced')

## Use resampling to create class balance
# Identify data points from majority and minority classes
data_majority = data[data["verified_status"] == "not verified"]
data_minority = data[data["verified_status"] == "verified"]

# Upsample the minority class (which is "verified")
data_minority_upsampled = resample(data_minority,
                                 replace=True,                 # to sample with replacement
                                 n_samples=len(data_majority), # to match majority class
                                 random_state=0)               # to create reproducible results

# Combine majority class with upsampled minority class
data_upsampled = pd.concat([data_majority, data_minority_upsampled]).reset_index(drop=True)

# Display new class counts
print('Class distribution after resampling: ', data_upsampled["verified_status"].value_counts())


## Get the video_transcription_text by class
data_upsampled[["verified_status", "video_transcription_text"]].groupby(by="verified_status")[["video_transcription_text"]].agg(func=lambda array: np.mean([len(text) for text in array]))


## Extract the video_transcription_text  and add it to a new column in a dataframe as this could be used as a potential feature in the model
data_upsampled["text_length"] = data_upsampled["video_transcription_text"].apply(func=lambda text: len(text))
## Visualize this new feature in a plot, where the data of versified and not_verified is presented seperatly
sns.histplot(data=data_upsampled, stat="count", multiple="stack", x="text_length", kde=False, palette="pastel",
             hue="verified_status", element="bars", legend=True)
plt.xlabel("video_transcription_text length (number of characters)")
plt.ylabel("Count")
plt.title("Distribution of video_transcription_text length for videos posted by verified accounts and videos posted by unverified accounts")
plt.show()


### Examine Correlations
# Code a correlation matrix to help determine most correlated variables
print(data_upsampled.corr(numeric_only=True))
# visualize the data in a correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(
    data_upsampled[["video_duration_sec", "claim_status", "author_ban_status", "video_view_count",
                    "video_like_count", "video_share_count", "video_download_count", "video_comment_count", "text_length"]]
    .corr(numeric_only=True),
    annot=True,
    cmap="crest")
plt.title("Heatmap of the dataset")
plt.show()


### Select Variables
# Severe multicollinearity among features is a problem, and it should be considered when selecting the variables to include in the model
# video_view_count and video_like_count show high correlation (0.86), thus one of these features should be excluded. video_like_count will be excluded.
# This variable also has a high correlation with video_download_Count and video_share_count.


# Select features, besides video_like_count also the # and video_id were not included in X since these are not expected to have any predictive power
X = data_upsampled[["video_duration_sec", "claim_status", "author_ban_status", "video_view_count", "video_share_count", "video_download_count", "video_comment_count"]]
# Display first few rows of features dataframe
X.head()

# Select outcome/target variable
y = data_upsampled["verified_status"]

### Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)
# Get shape of each training and testing set
print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

### encode non-numeric variables/features through one-hot encoding
# Check data types
print(X_train.dtypes)

#get unitque values for non-numerical features
# Get unique values in `claim_status`
print(X_train["claim_status"].unique())
# Get unique values in `author_ban_status`
print(X_train["author_ban_status"].unique())

X_train_to_encode = X_train[["claim_status", "author_ban_status"]]

# Set up an encoder for one-hot encoding the categorical features
X_encoder = OneHotEncoder(drop='first', sparse_output=False)

# Fit and transform the training features using the encoder
X_train_encoded = X_encoder.fit_transform(X_train_to_encode)

# Get feature names from encoder
names = X_encoder.get_feature_names_out()

# Place encoded training features (which is currently an array) into a dataframe
X_train_encoded_df = pd.DataFrame(data=X_train_encoded, columns=names)

# Display first few rows of encoded training features
print(X_train_encoded_df.head())

# Concatenate `X_train` and `X_train_encoded_df` to form the final dataframe for training data (`X_train_final`)
# Drop columns `claim_status` and `author_ban_status` since these will be replaced by the one-hot encoding
X_train_final = pd.concat([X_train.drop(columns=["claim_status", "author_ban_status"]).reset_index(drop=True), X_train_encoded_df], axis=1)

# Display first few rows
print(X_train_final.head())

## The target column is also not numeric and will have to be coverted again using the one-hot encoder
# Set up an encoder for one-hot encoding the categorical outcome variable
y_encoder = OneHotEncoder(drop='first', sparse_output=False)
#   - Adjusting the shape of `y_train` before passing into `.fit_transform()`, since it takes in 2D array
#   - Using `.ravel()` to flatten the array returned by `.fit_transform()`, so that it can be used later to train the model
y_train_final = y_encoder.fit_transform(y_train.values.reshape(-1, 1)).ravel()


### Model Building ###
###
# Construct a logistic regression model and fit it to the training set
log_clf = LogisticRegression(random_state=0, max_iter=800).fit(X_train_final, y_train_final)

## To evaluate the model X_test non-numeric varibales also need to be encoded. In the same as X_train.
X_test_to_encode = X_test[["claim_status", "author_ban_status"]]
# Transform the testing features using the encoder
X_test_encoded = X_encoder.transform(X_test_to_encode)
# Place encoded testing features (which is currently an array) into a dataframe
X_test_encoded_df = pd.DataFrame(data=X_test_encoded, columns=names)

# Concatenate `X_test` and `X_test_encoded_df` to form the final dataframe for training data (`X_test_final`)
X_test_final = pd.concat([X_test.drop(columns=["claim_status", "author_ban_status"]).reset_index(drop=True), X_test_encoded_df], axis=1)

# Display first few rows
print(X_test_final.head())



### Test the model
y_pred = log_clf.predict(X_test_final)
print(y_pred)

# Encode the testing outcome variable
y_test_final = y_encoder.transform(y_test.values.reshape(-1, 1)).ravel()


# Compute values for confusion matrix
log_cm = confusion_matrix(y_test_final, y_pred, labels=log_clf.classes_)
# Create display of confusion matrix
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=log_clf.classes_)
# Plot confusion matrix
log_disp.plot()
# Display plot
plt.show()


print(classification_report(y_test_final, y_pred))
print('The metrics from the verified class vs not verified are different')

## Analyse the coeficients (log-odds)
model_coef = pd.DataFrame(data={"Feature Name":log_clf.feature_names_in_, "Model Coefficient":log_clf.coef_[0]})
print(model_coef)

print('Each additional second of the video is associated with 0.009 increase in the log-odds of the user having a verified status.')
print('The recall for predicting the verified status based on the videos features was 84% however the precision and overall accuracy is rather low 61% and 65% respectively')


#########################
#### Part3: ML Model ####
#########################
# Objective predict whether a TikTok video presents a "claim" or presents an "opinion".
# This can help with the identification of videos that violate the platforms terms fo service
# videos that present claims are much more likely to violate the terms of service thus identifying those before further analysis and reviewing would be helpful

# The claim_status feature will be the target variable and this can take two values: "opinion” or “claim.” --> the model will be predicting a binary class

# Selecting the evaluation metric
# In this scenario it is less detrimental to predict false positives when a mistake is made than predict false negatives (classify videos as opinions when in fact they are claims,
# because that can mean that a video that breaks the terms of serviced is not identified).
# Because it is more important to minimize the false negatives recall is the best metric for model evaluation

# ~20,000 videos, the data will be split in 60/20/20 (train,validation, test)
# On part 2, the outliers were evaluated and dealt with. However, because tree-based models are robust to outliers will continue on this section with
# the initial data after removing the missing data.

print(data_3.info(), data_3.describe())

print(data_3["claim_status"].value_counts(normalize=True))
print('The data is balanced between the two classes')

# Feature engineering
# Calculate a new feature: length if the video transcript
data_3['text_length'] = data_3['video_transcription_text'].str.len()

print(data_3.groupby('claim_status')['text_length'].mean())
# Claim videos on average have more characters (~95 vs ~82)

sns.histplot(data=data_3, stat="count", multiple="dodge", x="text_length", kde=False, palette="pastel",
             hue="claim_status", element="bars", legend=True)
plt.xlabel("video_transcription_text length (number of characters)")
plt.ylabel("Count")
plt.title("Distribution of video_transcription_text length for videos posted by claim status")
plt.show()

# The length of the videos for both claims and opinions have an approximately normal distributions with a slight right tailing.

## Feature transformation and selection

X = data_3.copy()
# Drop '#' and 'video' idea as these columns have no predictive power.
X = X.drop(['#', 'video_id'], axis=1)

# Encode the target variable
# 'opinion' : 0 and 'claim' : 1
X['claim_status'] = X['claim_status'].replace({'opinion': 0, 'claim': 1})
# dummy encode the categorical features: 'verified_status', 'author_ban_status'
X = pd.get_dummies(X,
                   columns=['verified_status', 'author_ban_status'],
                   drop_first=True)

# isolate target variable
y = X['claim_status']

# isolate feature columns
X = X.drop(['claim_status'], axis=1)

# Split the data set to a final ratio of 60/20/20 (train/validation/test sets)
X_tr, X_test, y_tr, y_test = train_test_split(X, y, test_size=0.2, random_state=0)
X_train, X_val, y_train, y_val = train_test_split(X_tr, y_tr, test_size=0.25, random_state=0)
print(X_train.shape, X_val.shape, y_train.shape, y_val.shape, X_test.shape, y_test.shape)



# Set up a `CountVectorizer` object, to convert a collection of text (video_transcript_text variable) to a matrix of token counts
count_vec = CountVectorizer(ngram_range=(2, 3),
                            max_features=15,
                            stop_words='english')
# fit the vectorizer to the training data (generates 2-grams and 3-grams) and transform it meaning counting the number of occurrences. Only the training data should be fitted
count_data = count_vec.fit_transform(X_train['video_transcription_text']).toarray()

count_df = pd.DataFrame(data=count_data, columns=count_vec.get_feature_names_out())

# Concatenate `X_train` and `count_df` to form the final dataframe for training data (`X_train_final`)
# Note: Using `.reset_index(drop=True)` to reset the index in X_train after dropping `video_transcription_text`,
# so that the indices align with those in `X_train` and `count_df`
X_train_final = pd.concat([X_train.drop(columns=['video_transcription_text']).reset_index(drop=True), count_df], axis=1)
print(X_train_final.head())


# Transform the validation and test datasets. Get the n=gram counts for these. Note: the vectorizer is not refitted two these two data sets instead is only used for transforming the datasets.
# Meaning, the transcriptions of the videos in the validation data are only being checked against the n-grams found in the training data.

validation_count_data = count_vec.transform(X_val['video_transcription_text']).toarray()
validation_count_df = pd.DataFrame(data=validation_count_data, columns=count_vec.get_feature_names_out())
X_val_final = pd.concat([X_val.drop(columns=['video_transcription_text']).reset_index(drop=True), validation_count_df], axis=1)
print(X_val_final.head())


test_count_data = count_vec.transform(X_test['video_transcription_text']).toarray()
test_count_df = pd.DataFrame(data=test_count_data, columns=count_vec.get_feature_names_out())
X_test_final = pd.concat([X_test.drop(columns=['video_transcription_text']
                                      ).reset_index(drop=True), test_count_df], axis=1)
print(X_test_final.head())


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
rf = RandomForestClassifier(random_state=0)
cv_params = {'max_depth': [5, 7, None],
             'max_features': [0.3, 0.6],
            #  'max_features': 'auto'
             'max_samples': [0.7],
             'min_samples_leaf': [1,2],
             'min_samples_split': [2,3],
             'n_estimators': [75,100,200],
             }

scoring = ['accuracy', 'precision', 'recall', 'f1']
rf_cv = GridSearchCV(rf, cv_params, scoring=scoring, cv=5, refit='recall')

start = datetime.now()
rf_cv.fit(X_train_final, y_train)
print(datetime.now() - start)
write_pickle(path, rf_cv, 'rf_tt')

#rf_cv = read_pickle(path, 'rf_tt')


print(rf_cv.best_score_)
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


results = make_results('RF CV', rf_cv, 'recall')
print(results)


rf_preds = rf_cv.best_estimator_.predict(X_val_final)

# Use the function get_test_scores() output the scores of the model on the test dataset.

def get_test_scores(model_name:str, preds, y_test_data):
    '''
    Generate a table of test scores.

    In:
    model_name (string): how the model will be named in the output table
    preds: numpy array of test predictions
    y_test_data: numpy array of y_test data

    Out:
    table: a pandas df of precision, recall, f1, and accuracy scores for the model
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

rf_val_scores = get_test_scores('RF val', rf_preds, y_val)
results = pd.concat([results, rf_val_scores], axis=0)
print(results)

cm00 = confusion_matrix(y_val, rf_preds)

## Build a XGBoost model

Instantiate the XGBoost classifier
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
xgb_cv.fit(X_train_final, y_train)
print(datetime.now() - start)
write_pickle(path, xgb_cv, 'xgb_tt')

#xgb_cv = read_pickle(path, 'xgb_tt')

print(xgb_cv.best_score_)
print(xgb_cv.best_params_)


xgb_cv_results = make_results ('XGBoost CV', xgb_cv, 'recall')
results = pd.concat([results, xgb_cv_results], axis=0)
print(results)


xgb_preds = xgb_cv.best_estimator_.predict(X_val_final)
xgb_val_scores = get_test_scores('XGBoost val', xgb_preds, y_val)
results = pd.concat([results, xgb_val_scores], axis=0)
print(results)
cm01 = confusion_matrix(y_val, xgb_preds)



# Display the confusion matrices for both models side by side (RF vs XBGoost)

# Set formatting and styling options for the confusion matrices
title_size = 16
plt.rcParams.update({'font.size':16})
display_labels = ["Opinion", "Claim"]
colorbar = False

f, axes = plt.subplots(1, 2, figsize=(10, 8))
# Plot the first confusion matrix (Model 1) at position (0, 0)
axes[0].set_title("Random Forest", size=title_size)
ConfusionMatrixDisplay(confusion_matrix=cm00, display_labels=display_labels).plot(
    include_values=True, ax=axes[0], colorbar=colorbar)


# Plot the second confusion matrix (Model 2) at position (0, 1)
axes[1].set_title("XGBoost", size=title_size)
ConfusionMatrixDisplay(confusion_matrix=cm01, display_labels=display_labels).plot(
    include_values=True, ax=axes[1], colorbar=colorbar)

# Remove x and y-axis labels and ticks
axes[1].yaxis.set_ticklabels(['', ''])
axes[1].set_ylabel('')
axes[1].tick_params(axis='y', which='both',left=False)

# Set the figure's suptitle and display the plot
f.suptitle("Multiple Confusion Matrices: RF vs XGBoost", size=title_size, y=0.93)

plt.show()

# Save the figure as a PDF file
f.savefig("multiple_confusion_matrices.pdf", bbox_inches='tight')


## The metrics for both models are quite good however the recall is slightly better for the random forest model.
## In agreement most errors for XGBoost are false negatives, while random forest has less errors and they are equally distributed.
## Identifying claims (positves) was a priority thus the Random forest model should be selected and will be used to predict on the test data.

y_pred_test_rf = rf_cv.best_estimator_.predict(X_test_final)

log_cm = confusion_matrix(y_test,y_pred_test_rf)
log_disp = ConfusionMatrixDisplay(confusion_matrix=log_cm, display_labels=display_labels)
log_disp.plot()
plt.title('Random forest - test set');
plt.show()

# Feature importance plot for the RF model
importances = rf_cv.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test_final.columns)

fig, ax = plt.subplots()
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
fig.tight_layout()

## The most predictive features are all related to engagement levels. This is also in aggrement with the prior EDA (part 2)

## The RF model performed well both on the validation as well as the test set. All metrics were consistently high. Given the currant model performance adding new features
# does not seem necessary. However, it could be useful and informative to have the information on how many times a video was reported
# as well as having the total number of user reports for all videos posted by each author
