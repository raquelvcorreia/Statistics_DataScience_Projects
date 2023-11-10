from collections import Counter
from matplotlib import pyplot
from numpy import where
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from imblearn.under_sampling import RandomUnderSampler
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTEENN


# define dataset, make_classification() scikit-learn function can be used to define a synthetic dataset with
# a desired class imbalance. The “weights” argument specifies the ratio of examples in the negative class,
# e.g. [0.99, 0.01] means that 99 percent of the examples will belong to the majority class, and the remaining
# 1 percent will belong to the minority class
X, y = make_classification(n_samples=1000, n_features=2, n_redundant=0, n_clusters_per_class=1, weights=[0.99], flip_y=0, random_state=0)

# summarize class distribution
counter = Counter(y)
print(counter)

# define undersample strategy
undersample = RandomUnderSampler(sampling_strategy=0.5, random_state=0)
# fit and apply the transform
X_under, y_under = undersample.fit_resample(X, y)
# summarize class distribution
print(Counter(y_under))


# define oversample strategy (1:2)
oversample = SMOTE(sampling_strategy=0.5, random_state=0)
# fit and apply the transform
X_over, y_over = oversample.fit_resample(X, y)
# summarize class distribution
print(Counter(y_over))


# define smoteen sampling strategy
sample_combine = SMOTEENN(sampling_strategy=0.5, random_state=0)
# fit and apply the transform
X_combine, y_combine = sample_combine.fit_resample(X, y)
# summarize class distribution
print(Counter(y_combine))


# scatter plot of examples by class label
for label, _ in counter.items():
 row_ix = where(y_combine == label)[0]
 pyplot.scatter(X_combine[row_ix, 0], X_combine[row_ix, 1], label=str(label))
pyplot.legend()
pyplot.show()




# split into train/test sets with same class ratio (strtified=y), test and train data will be 50% each in this case.
trainX, testX, trainy, testy = train_test_split(X, y, test_size=0.5, stratify=y, random_state=0)
# define model
model = LogisticRegression(solver='liblinear')
# fit model
model.fit(trainX, trainy)
# predict on test set
yhat = model.predict(testX)
# evaluate predictions
print('Accuracy: %.3f' % accuracy_score(testy, yhat))
print('Precision: %.3f' % precision_score(testy, yhat))
print('Recall: %.3f' % recall_score(testy, yhat))
print('F-measure: %.3f' % f1_score(testy, yhat))


###this did not work very well probably because there is a very limnited numberr off samples
# split into train/test sets with same class ratio (stratified=y --> stratify based on data set y), test and train data will be 50% each in this case.
trainX_under, testX_under, trainy_under, testy_under = train_test_split(X_under, y_under, test_size=0.5, stratify=y_under, random_state=0)
# define model
model_under = LogisticRegression(solver='liblinear')
# fit model
model_under.fit(trainX_under, trainy_under)
# predict on test set
yhat_under = model_under.predict(testX_under)
# evaluate predictions
print('Accuracy (undersampling majority): %.3f' % accuracy_score(testy_under, yhat_under))
print('Precision (undersampling majority): %.3f' % precision_score(testy_under, yhat_under))
print('Recall (undersampling majority): %.3f' % recall_score(testy_under, yhat_under))
print('F-measure (undersampling majority): %.3f' % f1_score(testy_under, yhat_under))


###oversampling the minority class
# split into train/test sets with same class ratio (stratified=y --> stratify based on data set y), test and train data will be 50% each in this case.
trainX_over, testX_over, trainy_over, testy_over = train_test_split(X_over, y_over, test_size=0.5, stratify=y_over, random_state=0)
# define model
model_over = LogisticRegression(solver='liblinear')
# fit model
model_over.fit(trainX_over, trainy_over)
# predict on test set
yhat_over = model_over.predict(testX_over)
# evaluate predictions
print('Accuracy (oversampling minority): %.3f' % accuracy_score(testy_over, yhat_over))
print('Precision (oversampling minority): %.3f' % precision_score(testy_over, yhat_over))
print('Recall (oversampling minority): %.3f' % recall_score(testy_over, yhat_over))
print('F-measure (oversampling minority): %.3f' % f1_score(testy_over, yhat_over))


###combine undersamplying of the majority class and oversampling the minority class - SMOTEENN
# split into train/test sets with same class ratio (stratified=y --> stratify based on data set y), test and train data will be 50% each in this case.
trainX_combine, testX_combine, trainy_combine, testy_combine = train_test_split(X_combine, y_combine, test_size=0.5, stratify=y_combine, random_state=0)
# define model
model_combine = LogisticRegression(solver='liblinear')
# fit model
model_combine.fit(trainX_combine, trainy_combine)
# predict on test set
yhat_combine = model_combine.predict(testX_combine)
# evaluate predictions
print('Accuracy (SMOTEENN): %.3f' % accuracy_score(testy_combine, yhat_combine))
print('Precision (SMOTEENN): %.3f' % precision_score(testy_combine, yhat_combine))
print('Recall (SMOTEENN): %.3f' % recall_score(testy_combine, yhat_combine))
print('F-measure (SMOTEENN): %.3f' % f1_score(testy_combine, yhat_combine))



# define model with a class weight argument -->cost sensitive algorithm. It will define a penalty verse of the distribution penalizing more errors with the minority class
model_cost = LogisticRegression(solver='liblinear', class_weight='balanced')
# fit model
model_cost.fit(trainX, trainy)
# predict on test set
yhat_cost = model_cost.predict(testX)
# evaluate predictions
print('F-Measure (cost_function): %.3f' % f1_score(testy, yhat_cost))
