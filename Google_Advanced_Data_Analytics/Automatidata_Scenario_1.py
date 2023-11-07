### Questions being addressed
# New York City Taxi and Limousine Commission (TLC) dataset, data on 22699 rides with information on pickup and dropoff times and locations,
# number of passenger, trip distance, fares, payment, fare and tips information
# data overview,  cleanup and preparation as well as EDA will be performed before modeling

## Part 1: Statistical analysis looking at the payment method.

## Part 2: Linear regression to predict the fare amount.

## Part 3: Build a model to predict if a client/rider will leave a tip



## Import necessary packages
from datetime import datetime
import numpy as np
import pandas as pd
import os
np.warnings.filterwarnings('ignore')

import matplotlib.pyplot as plt
import seaborn as sns

from scipy import stats


# Pickle allows to save your ML models, to minimise lengthy re-training and allow you to share, commit, and re-load pre-trained machine learning models.
# Once the model has been trained the fit and the write to pickle instructions should be commented out and read one uncommented.
import pickle



# Import packages for data preprocessing
from sklearn.preprocessing import StandardScaler

# Import packages for regression and metrics
import sklearn.metrics as metrics
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score, recall_score, precision_score, accuracy_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error,r2_score,mean_squared_error

# For modeling
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

# This is the function that helps plot feature importance
from xgboost import plot_importance



## In order to be able to evaluate more columns when using the IDE
desired_width=320
pd.set_option('display.width', desired_width)
np.set_printoptions(linewidth=desired_width)
pd.set_option('display.max_columns',23)

# or alternatively use this:
#pd.set_option('display.max_columns', None)

#importa dataset into a dataframe
taxi_data = pd.read_csv("2017_Yellow_Taxi_Trip_Data.csv")



# Exploratory Analysis, have an idea of how the data looks like and get general statistics for all features
print(taxi_data.head(10))
print(taxi_data.shape)
print(taxi_data.info())
## Keep taxi_data as the original dataframe create a copy of it to use in further analysis and transformations, if needed we can always revert to the original df
df = taxi_data.copy()
print(taxi_data.describe(include = 'all'))

#  look at the average fare amount for each payment type
print(taxi_data.groupby('payment_type', as_index = False)['fare_amount'].mean())




############################
### Part 1               ###
### Statistical Analysis ###
############################

## 1 - credit card payment
## 2 - cash payment
### the fees paid by credit card are on average higher, however is this difference statistically signficant

## Hypothesis testing
## 𝐻0:There is no difference in the average fare amount between customers who use credit cards and customers who use cash.
## 𝐻𝐴: There is a difference in the average fare amount between customers who use credit cards and customers who use cash.

# set significance level
significance_level = 0.05

## create data frames for each sample being compared
taxi_data_creditcard = taxi_data[taxi_data['payment_type'] == 1]
taxi_data_cash = taxi_data[taxi_data['payment_type'] == 2]

# p-value calculation
tstat, pvalue = stats.ttest_ind(taxi_data_creditcard['fare_amount'], b=taxi_data_cash['fare_amount'], equal_var=False)
print(tstat)

print(pvalue)


## The p-value is significantly smaller than the significance level o f5% thus the null hypothesis is rejected. The difference in the average fare amount paid
## between customers who use a credit card and the ones that pay in cash is statistically different.



############################
### Part 2               ###
### Linear Regression    ###
############################


print(df.info())

# Check for missing data and duplicates using .isna() and .drop_duplicates()
print('Shape of dataframe:', df.shape)
print('Shape of dataframe with duplicates dropped:', df.drop_duplicates().shape)

# total number of missing values
print('The total number of missing values is: ', df.isna().sum().sum())
print(df.describe())

# There are some outliers, and some variables are mostly constant through the data such as mta_tax and improvement_surcharge which likely will not be very predictive

## Convert the pickup and dropoff columns to datatime
# Check the format of the data
df['tpep_dropoff_datetime'][0]

# Convert datetime columns to datetime
# Display data types of `tpep_pickup_datetime`, `tpep_dropoff_datetime`
print('Data type of tpep_pickup_datetime:', df['tpep_pickup_datetime'].dtype)
print('Data type of tpep_dropoff_datetime:', df['tpep_dropoff_datetime'].dtype)

# Convert `tpep_pickup_datetime` to datetime format
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')

# Convert `tpep_dropoff_datetime` to datetime format
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')

# Display data types of `tpep_pickup_datetime`, `tpep_dropoff_datetime`
print('Data type of tpep_pickup_datetime:', df['tpep_pickup_datetime'].dtype)
print('Data type of tpep_dropoff_datetime:', df['tpep_dropoff_datetime'].dtype)

# Create `duration` column
df['duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime'])/np.timedelta64(1,'m')


### Dealing with outliers
## Not all the columns will be used in the model. Thus not all columns need to be checked for outliers. The most important are:
# trip_distance, fare_amount and duration. Use boxplot to visualize the data for these three columns.

fig, axes = plt.subplots(1, 3, figsize=(15, 2))
fig.suptitle('Boxplots for outlier detection')
sns.boxplot(ax=axes[0], x=df['trip_distance'])
sns.boxplot(ax=axes[1], x=df['fare_amount'])
sns.boxplot(ax=axes[2], x=df['duration'])
plt.show()

## All 3 columns contain outliers.
# trip_distance distribution seems reasonable and will not be changed.
# However, both fare_amount and duration have significant outliers at the higher end.

## trip_distance outliers
print(sorted(set(df['trip_distance']))[:10])

# How many trips have distance '0' (for instances if a passenger summoned a taxi and then ended up not using it)
sum(df['trip_distance']==0)
# only 148 out of 22699 rides have a distance 0 these are not likely to have a significant impact. The higher end ones seem reasonable with the distance between
# staten Island and the northern end of Manhattan.


## fare_amount outliers
# The range is high: -120 to 999.99.
# While '0' would be acceptable if a taxi driver logged a trip that was than cancelled and did not happen but negative values can not be explained
# The other extreme seems unlikely and could compromise the model. Thus, these will be capped at a maximum value based on intuition and statistics.
# Calculate the interquartile range for the fare_amount(iqr = q3-q1)
q1_fare, q3_fare = np.percentile(df['fare_amount'], [25, 75])
iqr_fare = q3_fare - q1_fare
# calculate the higher fence using the standard formula (the outliers of concern are on the higher end of the data)
print('The IQR for the fare_amount column is: ', iqr_fare)
print('The Q3 for the fare_amount column is: ', q3_fare)
print('The upper threshold considering the standard IQR factor of 1.5 is: ', (q3_fare + (1.5*iqr_fare)))
n_above = (df['fare_amount'] > 26.50).sum()
print('Number of fares above the higher fence ', n_above)
# the higher fence is $26.5. Intuitively capping the fare_amount to that value does not seem appropriate. Additionally, 2063 rides are above $26.5.
#using an iqr_factor of 6 instead of 1.5.
n_above_2 = (df['fare_amount'] > 62.50).sum()
print('Number of fares above the higher fence considering an IQR of 6 ', n_above_2)
# using IQR factor of 6 instead of 8 results in a cap if $62.50, seems more appropriate and only 82 rides are above that.


# Function to impute upper-limit in specified columns based on their interquartile range, and reassigning the minimum value to '0'
def outlier_imputer(column_list, iqr_factor):
    '''
    Impute upper-limit values in specified columns based on their interquartile range.

    Arguments:
        column_list: A list of columns to iterate over
        iqr_factor: A number representing x in the formula:
                    Q3 + (x * IQR). Used to determine maximum threshold,
                    beyond which a point is considered an outlier.

    The IQR is computed for each column in column_list and values exceeding
    the upper threshold for each column are imputed with the upper threshold value.

    0 is imputed to any negative value.
    '''
    for col in column_list:
        # Reassign minimum to zero
        df.loc[df[col] < 0, col] = 0

        # Calculate upper threshold
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1
        upper_threshold = q3 + (iqr_factor * iqr)
        print(col)
        print('q3:', q3)
        print('upper_threshold:', upper_threshold)

        # Reassign values > threshold to threshold
        df.loc[df[col] > upper_threshold, col] = upper_threshold
        print(df[col].describe())



print(outlier_imputer(['fare_amount'],6))


## duration outliers
print(df['duration'].describe())
# 0 should be imputed to any negative numbers (there are no negative durations)
# duration also has some high values outliers. The max value is 1439.55 eventhough the Q75 is ~18, Impute a cap similar to what was done for the fare_amount.
# Will be using the same function.
print(outlier_imputer(['duration'],6))



#### Feature Engineering  ####
## Create mean_distance column that captures the mean distance for each group of trips that share pickup and dropoff points
# When diploying the model the duration of a trip can not be known until it finishes so this can not be a feature used to train the model.
# However, one can use statistics of the trips already performed to generalize about the ones to come.
# Create `pickup_dropoff` column
df['pickup_dropoff'] = df['PULocationID'].astype(str) + ' ' + df['DOLocationID'].astype(str)
print(df['pickup_dropoff'].head(2))

# create a new dataframe using groupby to get the mean of all the trips with the same pickup and dropoff locations
grouped = df.groupby('pickup_dropoff').mean(numeric_only=True)[['trip_distance']]
print(grouped[:5])

# 1. Convert `grouped` to a dictionary
grouped_dict = grouped.to_dict()

# 2. Reassign to only contain the inner dictionary
grouped_dict = grouped_dict['trip_distance']
print(grouped_dict)

# create a mena_distance column in df, that is a copy of the pickup_dropoff column. Using map replace the trip information by its mean duration.
# 1. Create a mean_distance column that is a copy of the pickup_dropoff helper column
df['mean_distance'] = df['pickup_dropoff']

# 2. Map `grouped_dict` to the `mean_distance` column
df['mean_distance'] = df['mean_distance'].map(grouped_dict)


## Create a mean durantion column following the same process
grouped_duration= df.groupby('pickup_dropoff').mean(numeric_only=True)[['duration']]
print(grouped_duration[:5])

# Create a dictionary where keys are unique pickup_dropoffs and values are
# mean trip duration for all trips with those pickup_dropoff combos
grouped_dict_duration = grouped_duration.to_dict()
grouped_dict_duration = grouped_dict_duration['duration']
df['mean_duration'] = df['pickup_dropoff']
df['mean_duration'] = df['mean_duration'].map(grouped_dict_duration)

## Create day and month columns extracting the information from the tpep_pickup_datetime column
df['day'] = df['tpep_pickup_datetime'].dt.day_name().str.lower()
df['month'] = df['tpep_pickup_datetime'].dt.strftime('%b').str.lower()

## create a binary rush hour column. If the ride was during rush hour and a 0 if it was not.
# Any weekday (not Saturday or Sunday) AND
# Either from 06:00–10:00 or from 16:00–20:00
df['rush_hour'] = df['tpep_pickup_datetime'].dt.hour
df.loc[df['day'].isin(['saturday', 'sunday']), 'rush_hour'] = 0

# define a function to assign 0 or 1 based on the pickup hour.
def rush_hourizer(hour):
    if 6 <= hour['rush_hour'] < 10:
        val = 1
    elif 16 <= hour['rush_hour'] < 20:
        val = 1
    else:
        val = 0
    return val


# Apply the `rush_hourizer()` function to the new column
df.loc[(df.day != 'saturday') & (df.day != 'sunday'), 'rush_hour'] = df.apply(rush_hourizer, axis=1)
print(df.head())


#### visualize the relationship between mean_duration and fare_amount
sns.set(style='whitegrid')
f = plt.figure()
f.set_figwidth(5)
f.set_figheight(5)
sns.regplot(x=df['mean_duration'], y=df['fare_amount'],
            scatter_kws={'alpha':0.5, 's':5},
            line_kws={'color':'red'})
plt.ylim(0, 70)
plt.xlim(0, 70)
plt.title('Mean duration x fare amount')
plt.show()

## The mean_duration variable correlates with the fare_amount. However there are 2 horizontal lines: one at around 62 and another arrond 52.
# 62.50 is the value imputed to all outliers hence the line.
# Evaluate more closely all the fare_amounts higher than 50
print(df[df['fare_amount'] > 50]['fare_amount'].value_counts().head())

# There is an unusual amount of trips with exact the same fare_amount $52 --> 514 trips.
print(df[df['fare_amount']==52].head(30))

## all trip seem to have the sam RatecodeID and either originate or end in location 132. Also, there are several similar toll amounts.
# RatecodeID corresponds to trips between JFK and Manhattan and quick seach shows that in 2017 there was a taxi flat rate for these trips $52.
# RatecodeID is known from the data dictionary, the values for this rate code can be imputed back into the data after the model makes its predictions.
# This way you know that those data points will always be correct.

##### Isolate Modeling Variables
## Drop all redundant and irrelevant features. As well as the ones that will not be avaialble in a deployed enviornment.
## create a df2 which is a copy of df to which all changes will be applied
df2 = df.copy()

df2 = df2.drop(['Unnamed: 0', 'tpep_dropoff_datetime', 'tpep_pickup_datetime',
               'trip_distance', 'RatecodeID', 'store_and_fwd_flag', 'PULocationID', 'DOLocationID',
               'payment_type', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount', 'improvement_surcharge',
               'total_amount', 'tpep_dropoff_datetime', 'tpep_pickup_datetime', 'duration',
               'pickup_dropoff', 'day', 'month'
               ], axis=1)

print(df2.info())

# Create a pairplot to visualize pairwise relationships between variables in the data
sns.pairplot(df2[['fare_amount', 'mean_duration', 'mean_distance']],
             plot_kws={'alpha':0.4, 'size':5},
             )
plt.show()



#### Identify correlations  ####
# Create correlation matrix containing pairwise correlation of columns, using pearson correlation coefficient
df2.corr(method='pearson')

# Create correlation heatmap
plt.figure(figsize=(6,4))
sns.heatmap(df2.corr(method='pearson'), annot=True, cmap='Reds')
plt.title('Correlation heatmap',
          fontsize=18)
plt.show()

# mean_duration and mean_distance are both highly correlated with fare_amount
# They're also both correlated with each other, with a Pearson correlation of 0.87
# Highly correlated predictor variables can be detrimental for linear regression models
# when the intent is to be able to draw statistical inferences about hte data from the model.
# However correlated predictor variables can still be used to create an accurate predictor
# if the prediction is more important than using the model as a tool to learn about the data.
# The model aims at predicting the fare_amount thus the two variables can be used even if highly correlated.

# Remove the target column from the features
X = df2.drop(columns=['fare_amount'])

# Set y variable (target)
y = df2[['fare_amount']]



### Pre-Process data ####
# Dummy encode categorical variables
# Convert VendorID to string
X['VendorID'] = X['VendorID'].astype(str)

# Get dummies
X = pd.get_dummies(X, drop_first=True)
X.head()

### Split data into training and test sets. Test set shoudl be 20% of total samples
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

### Standarize the data
scaler = StandardScaler().fit(X_train)
X_train_scaled = scaler.transform(X_train)
print('X_train scaled:', X_train_scaled)


### Fit your model to the training data
lr=LinearRegression()
lr.fit(X_train_scaled, y_train)


### Evaluate the model
# Evaluate the model performance on the training data
r_sq = lr.score(X_train_scaled, y_train)
print('Coefficient of determination:', r_sq)



y_pred_train = lr.predict(X_train_scaled)
print('R^2:', r2_score(y_train, y_pred_train))
print('MAE:', mean_absolute_error(y_train, y_pred_train))
print('MSE:', mean_squared_error(y_train, y_pred_train))
print('RMSE:',np.sqrt(mean_squared_error(y_train, y_pred_train)))

# Evaluate the model performance on the test data
# First the test data also has to be scaled
X_test_scaled = scaler.transform(X_test)
r_sq_test = lr.score(X_test_scaled, y_test)
print('Coefficient of determination:', r_sq_test)
y_pred_test = lr.predict(X_test_scaled)
print('R^2:', r2_score(y_test, y_pred_test))
print('MAE:', mean_absolute_error(y_test,y_pred_test))
print('MSE:', mean_squared_error(y_test, y_pred_test))
print('RMSE:',np.sqrt(mean_squared_error(y_test, y_pred_test)))


### Results
results = pd.DataFrame(data={'actual': y_test['fare_amount'],
                             'predicted': y_pred_test.ravel()})
results['residual'] = results['actual'] - results['predicted']
print(results.head())

# Create a scatterplot to visualize `predicted` over `actual`
fig, ax = plt.subplots(figsize=(6, 6))
sns.set(style='whitegrid')
sns.scatterplot(x='actual',
                y='predicted',
                data=results,
                s=20,
                alpha=0.5,
                ax=ax
)
plt.plot([0,60], [0,60], c='red', linewidth=2)
plt.title('Actual vs. predicted')
plt.show()

# Visualize the distribution of the `residuals`
sns.histplot(results['residual'], bins=np.arange(-15,15.5,0.5))
plt.title('Distribution of the residuals')
plt.xlabel('residual value')
plt.ylabel('count')
plt.show()

print(results['residual'].mean())
# The distribution of the residuals is normal and has a mean of -0.015.
# The residuals represent the variance in the outcome variable that is not explained by the model.



# Create a scatterplot of `residuals` over `predicted`
sns.scatterplot(x='predicted', y='residual', data=results)
plt.axhline(0, c='red')
plt.title('Scatterplot of residuals over predicted values')
plt.xlabel('predicted value')
plt.ylabel('residual value')
plt.show()

# The model's residuals are evenly distributed above and below zero,
# with the exception of the sloping lines from the upper-left corner to the lower-right corner,
# which correspond to the imputed maximum of $62.50 and the flat rate of $52 for JFK airport trips.


### Coeficients ###
coefficients = pd.DataFrame(lr.coef_, columns=X.columns)
print(coefficients)
#  for every +1 change in standard deviation, the fare amount increases by a mean of $7.13.


#  Translate the coeficients back to miles. For every 3.57 miles traveled, the fare increased by a mean of $7.13.
#  Or, reduced: for every 1 mile traveled, the fare increased by a mean of $2.00.
# 1. Calculate SD of `mean_distance` in X_train data
print(X_train['mean_distance'].std())
# 2. Divide the model coefficient by the standard deviation
print(7.133867 / X_train['mean_distance'].std())


# get the model predictions for the entire dataset.
# for some trips the fare can be imputed and does not need to be imputed (RatecodeID  = $52)

############################
### Part 3               ###
### Modeling             ###
############################


# Get the predictions for the full dataset

X_scaled = scaler.transform(X)
y_preds_full = lr.predict(X_scaled)

# New df with only the RatecodeID column
final_preds = df[['RatecodeID']].copy()

# Add a column containing all the model fare predictions (full dataset)
final_preds['y_preds_full'] = y_preds_full

# Impute a prediction of 52 at all rows where RatecodeID == 2
final_preds.loc[final_preds['RatecodeID']==2, 'y_preds_full'] = 52

# Check performance on full dataset
final_preds = final_preds['y_preds_full']
print('Performance of the linear regression method on the full dataset and after imputing the fare amounts on the preset trip amounts')
print('R^2:', r2_score(y, final_preds))
print('MAE:', mean_absolute_error(y, final_preds))
print('MSE:', mean_squared_error(y, final_preds))
print('RMSE:',np.sqrt(mean_squared_error(y, final_preds)))

# Combine mean columns with predictions column
nyc_preds_means = df[['mean_duration', 'mean_distance']].copy()
nyc_preds_means['predicted_fare'] = final_preds

print(nyc_preds_means.head())

# export this newly created dataframe, nyc_preds_means, to a csv file
nyc_preds_means.to_csv('nyc_preds_means.csv')


# merge this new dataframe containing the predictions with the original data set
df_w_preds = taxi_data.merge(nyc_preds_means,
                left_index=True,
                right_index=True)

print(df_w_preds.head())

# look at the mean by payment type
grouped_payment_type = df_w_preds.groupby('payment_type')['tip_amount'].mean()
print(grouped_payment_type)
print('Only riders using payment type 1 (credit card) show an tip amount average above 0, so we will focus the analysis/model on that group')
df_w_preds_cc = df_w_preds[df_w_preds['payment_type'] == 1]

# create a tip_percent column making sure to get the value rounded to 3 decimals
df_w_preds_cc['tip_percent'] = round(df_w_preds_cc['tip_amount'] / (df_w_preds_cc['total_amount'] - df_w_preds_cc['tip_amount']), 3)


# create a new column where 1 is assigned to rides with >=20% tip

df_w_preds_cc['generous'] = df_w_preds_cc['tip_percent']
df_w_preds_cc['generous'] = (df_w_preds_cc['generous'] >=0.2)
df_w_preds_cc['generous'] = df_w_preds_cc['generous'].astype(int)

### Part of the data cleaning had been previously done, but will be repeated since Part3, the original data was again loaded
# Convert pickup and dropoff cols to datetime
df_w_preds_cc['tpep_pickup_datetime'] = pd.to_datetime(df_w_preds_cc['tpep_pickup_datetime'], format='%m/%d/%Y %I:%M:%S %p')
df_w_preds_cc['tpep_dropoff_datetime'] = pd.to_datetime(df_w_preds_cc['tpep_dropoff_datetime'], format='%m/%d/%Y %I:%M:%S %p')

# create a columns with only the day of the pickup (this should be in lowercase)
df_w_preds_cc['day'] = df_w_preds_cc['tpep_pickup_datetime'].dt.day_name().str.lower()

# create 4 new columns: am_rush, daytime, pm_rush and nighttime
df_w_preds_cc['am_rush'] = df_w_preds_cc['tpep_pickup_datetime'].dt.hour
df_w_preds_cc['daytime'] = df_w_preds_cc['tpep_pickup_datetime'].dt.hour
df_w_preds_cc['pm_rush'] = df_w_preds_cc['tpep_pickup_datetime'].dt.hour
df_w_preds_cc['nighttime'] = df_w_preds_cc['tpep_pickup_datetime'].dt.hour


# create/define an am_rush function
def am_rush(hour):
    if 6 <= hour['am_rush'] < 10:
        val = 1
    else:
        val = 0
    return val

# .apply the function 'am_rush' to the 'am_rush' series
df_w_preds_cc['am_rush'] = df_w_preds_cc.apply(am_rush, axis=1)
df_w_preds_cc['am_rush'].head()

# do the same tranformations for the other 3 columns created

def pm_rush(hour):
    if 16 <= hour['pm_rush'] < 20:
        val = 1
    else:
        val = 0
    return val

def daytime(hour):
    if 10 <= hour['daytime'] < 16:
        val = 1
    else:
        val = 0
    return val

def nighttime(hour):
    if 20 <= hour['nighttime'] < 24:
        val = 1
    elif 0 <= hour['nighttime'] < 6:
        val = 1
    else:
        val = 0
    return val

df_w_preds_cc['pm_rush'] = df_w_preds_cc.apply(pm_rush, axis=1)
df_w_preds_cc['daytime'] = df_w_preds_cc.apply(daytime, axis=1)
df_w_preds_cc['nighttime'] = df_w_preds_cc.apply(nighttime, axis=1)


# create a month column
df_w_preds_cc['month'] = df_w_preds_cc['tpep_pickup_datetime'].dt.strftime('%b').str.lower()

# drop all redundant and irrelevant columns as well as those that have information that would not be available when the model is deployed.
# Keeping the target column (generous)

# columns to be dropped
print(df_w_preds_cc.info())

drop_col_lst = ['Unnamed: 0', 'tpep_pickup_datetime', 'tpep_dropoff_datetime', 'trip_distance', 'payment_type', 'store_and_fwd_flag', 'extra', 'mta_tax', 'tip_amount', 'tolls_amount','improvement_surcharge', 'total_amount', 'tip_percent']
df_w_preds_cc = df_w_preds_cc.drop(drop_col_lst, axis = 1)
print(df_w_preds_cc.head(10))


# Convert categorical columns to binary
cols_to_str = ['RatecodeID', 'PULocationID', 'DOLocationID', 'VendorID']
for col in cols_to_str:
    df_w_preds_cc[col] = df_w_preds_cc[col].astype('str')

df_w_preds_cc_2 = pd.get_dummies(df_w_preds_cc, drop_first=True)
print(df_w_preds_cc_2.info())

# Evaluate the classes balance before deciding on the metrics to be used.
# Get class balance of 'generous' col
print(df_w_preds_cc_2['generous'].value_counts(normalize=True))

## The dataset is nearly balanced with 53% vs 47% for each class. In this scenario both false positives and false negatives seem to be ultimately equivalent
# thus a metric that balances both should be preferable ==> F1 score

# split the data, 20% in the test set and statifying the data
y = df_w_preds_cc_2['generous']
X = df_w_preds_cc_2.drop('generous', axis = 1)
X_train, X_test, y_train, y_test = train_test_split(X, y , stratify = y, test_size = 0.2, random_state = 42)

## Random Forest for binary classification
# Use GridSearchCV to tune the model
# RF, 4-fold cross validation and setting refit to F1
# 1. Instantiate the random forest classifier
rf = RandomForestClassifier(random_state=42)

# 2. Create a dictionary of hyperparameters to tune
cv_params = {'max_depth': [10, 50],
             'max_features': [1.0],
             'max_samples': [.5,.9],
             'min_samples_leaf': [0.5,1],
             'min_samples_split': [2],
             'n_estimators': [50, 300]
             }
# 3. Define a set of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 4. Instantiate the GridSearchCV object
rf1 = GridSearchCV(rf, cv_params, scoring=scoring, cv=4, refit='f1')

start = datetime.now()
rf1.fit(X_train, y_train)
print(datetime.now() - start)

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

write_pickle(path, rf1, 'rf1_cv_automatidata')
#rf1 = read_pickle(path, 'rf1_cv_automatidata')

print(rf1.best_score_)

print(rf1.best_params_)



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


results = make_results('RF CV', rf1, 'f1')
print(results)


rf_preds = rf1.best_estimator_.predict(X_test)

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

rf_test_scores = get_test_scores('RF test', rf_preds, y_test)
results = pd.concat([results, rf_test_scores], axis=0)
print(results)

## Results on the test and train data sets were very close


## XGBoost for binary classification

# 1. Instantiate the XGBoost classifier
xgb = XGBClassifier(objective='binary:logistic', random_state=0)

# 2. Create a dictionary of hyperparameters to tune
cv_params = {'learning_rate': [0.1, 0.3],
             'max_depth': [4, 8],
             'min_child_weight': [2, 5],
             'n_estimators': [300, 500]
             }

# 3. Define a set of scoring metrics to capture
scoring = ['accuracy', 'precision', 'recall', 'f1']

# 4. Instantiate the GridSearchCV object
xgb1 = GridSearchCV(xgb, cv_params, scoring=scoring, cv=4, refit='f1')

start = datetime.now()
xgb1.fit(X_train, y_train)
print(datetime.now() - start)
write_pickle(path, xgb1, 'xgb1_cv_automatidata')

#xgb1 = read_pickle(path, 'xgb1_cv_automatidata')

print(xgb1.best_score_)
print(xgb1.best_params_)



xgb1_cv_results = make_results('XGB CV', xgb1, 'f1')
results = pd.concat([results, xgb1_cv_results], axis=0)
print(results)


# Get scores on test data
xgb_preds = xgb1.best_estimator_.predict(X_test)
xgb_test_scores = get_test_scores('XGB test', xgb_preds, y_test)
results = pd.concat([results, xgb_test_scores], axis=0)
print(results)

# The F1 score is ~0.01 lower for XGBoost than for the random forest model. Both models have very similar metrics with the ones for the RF being slightly better
# generate a confusion metric for the RF model and check the feature importances.

cm = confusion_matrix(y_test, rf_preds, labels=rf1.classes_)

disp = ConfusionMatrixDisplay(confusion_matrix=cm,
                             display_labels=rf1.classes_,
                             )
disp.plot(values_format='')
plt.grid(False)
plt.show()

# The rate of false positive higher than the one of false negatives.
# Type I errors are more common (Meaning that it will be more comment for drivers to expect a good tip and not have it than not expect it and actually get it)..

importances = rf1.best_estimator_.feature_importances_
rf_importances = pd.Series(importances, index=X_test.columns)
rf_importances = rf_importances.sort_values(ascending=False)[:15]

fig, ax = plt.subplots(figsize=(8,5))
rf_importances.plot.bar(ax=ax)
ax.set_title('Feature importances')
ax.set_ylabel('Mean decrease in impurity')
ax.grid(False)
fig.tight_layout()
plt.show()
