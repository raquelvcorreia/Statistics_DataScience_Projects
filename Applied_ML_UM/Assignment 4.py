import warnings
warnings.filterwarnings("ignore")

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.dummy import DummyClassifier
from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
from sklearn.model_selection import cross_validate, KFold
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
import time
from sklearn.preprocessing import MinMaxScaler
from imblearn.combine import SMOTEENN

np.random.seed(0)   # Do not change this value: required to be compatible with solutions generated by the autograder.

# read data and setting video ID as the index
train_data = pd.read_csv(r'C:\Users\raque\Documents\Applied ML in Python - MU\JupiterNotebookFiles\Files\home\jovyan\work\assignments\course3_assignment4\assets\train.csv').set_index('id')
features_names = ['title_word_count', 'document_entropy', 'freshness', 'easiness', 'fraction_stopword_presence', 'normalization_rate', 'speaker_speed', 'silent_period_rate']
X_train_data = train_data[features_names]
y_train_data = train_data['engagement']

X_test_data = pd.read_csv(r'C:\Users\raque\Documents\Applied ML in Python - MU\JupiterNotebookFiles\Files\home\jovyan\work\assignments\course3_assignment4\assets\test.csv').set_index('id')



# scale features using the minmax scaler
scaling = MinMaxScaler()
X_train_scaled = scaling.fit_transform(X_train_data)

X_test_scaled = scaling.transform(X_test_data)


#SMOTEEN (under and oversampling)
# define smoteen sampling strategy
sample_combine = SMOTEENN(sampling_strategy=0.5, random_state=0)
# fit and apply the transform
X_combine, y_combine = sample_combine.fit_resample(X_train_scaled, y_train_data)
# summarize class distribution
print(Counter(y_combine))

#check if there are NAs there are no NAs
check_na = train_data.isnull().values.any()

# evaluate if the dataset is imbalance
ss_counts = np.bincount(train_data.engagement)
print(ss_counts)
# alternatively summarize class distribution
counter = Counter(y_train_data)
print(counter)

def performance_graphics(thresholds, thresholds_roc, precision, recall, fp, tp, auc_score):
    close_default = np.argmin(np.abs(thresholds - 0.5))
    close_zero = np.argmin(np.abs(thresholds_roc))

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    ax1.plot(precision, recall, label="Precision Recall Curve")
    ax1.plot(precision[close_default],
             recall[close_default], 'o',
             c='r', markersize=10,
             label='threshold 0.5',
             fillstyle="none", mew=2)
    ax1.set_title("RF performance")
    ax1.set_xlabel("Precision")
    ax1.set_ylabel("Recall")
    ax1.legend(loc='best')

    ax2.plot(fp, tp, label="ROC curve")
    ax2.plot(fp[close_zero],
             tp[close_zero], 'o',
             c='r', markersize=10,
             label='threshold 0',
             fillstyle="none", mew=2)
    ax2.set_title(f"ROC performance: AUC Score {auc_score}")
    ax2.set_xlabel("False Positive Rate")
    ax2.set_ylabel("True Positive (Recall)")
    ax2.legend(loc='best')

    return plt.show()


def classifier_assessment(X, y, clf, show_graph=True):
    """
    args: X dataset of features
          y target set of values for classification

    returns:
        - the mean of the Logistic Regression Claffifier prediction scores taken from a
          5 fold cross validation on the dataset
    """
    kfold = KFold(n_splits=5)
    cross_val = cross_validate(clf, X, y, cv=kfold, return_estimator=True)
    mean_score = cross_val['test_score'].mean()

    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    clf.fit(X_train, y_train)
    proba_ = clf.predict_proba(X_test)[:, 1]

    precision, recall, thresholds = precision_recall_curve(y_test, proba_)
    fp, tp, thresholds_roc = roc_curve(y_test, proba_)

    auc_score = np.round(roc_auc_score(y_test, proba_), 4)

    # this will plot graph automatically
    if show_graph:
        performance_graphics(thresholds, thresholds_roc, precision, recall, fp, tp, auc_score)

    return auc_score, clf


def feat_importance_eval(clf, columns=X_train_data.columns):
    try:
        importances = clf.feature_importances_
    except:
        importances = clf.coef_[0]

    importance_dict = {}
    for importance, name in zip(importances, columns):
        if importance != 0:
            importance_dict[name] = importance

    importance_dict = sorted(importance_dict.items(), key=lambda x: -x[-1])[:10]
    return importance_dict


def plot_feature_importance(importance_dict):
    plt.figure(figsize=(15, 6));
    x_label = [x[0] for x in importance_dict]
    x_coords = list(range(len(importance_dict)))
    plt.bar(x=x_coords,
            width=0.5,
            height=[x[-1] for x in importance_dict],
            tick_label=x_label);

    ax = plt.gca()
    ax.set_xticklabels(rotation=45, labels=x_label)

    return plt.show()



## Logistic Regression
clf_log = LogisticRegression(random_state=0)
auc_score_log, clf_log = classifier_assessment(X_train_scaled, y_train_data, clf=clf_log)
print("Logistic Regression AUC score: ", auc_score_log)



pipe = Pipeline([('classifier' , LogisticRegression(random_state=1))])
param_grid = [
    {'classifier' : [LogisticRegression(random_state=1)],
     'classifier__penalty' : ['l1', 'l2'],
     'classifier__C' : np.logspace(-4, 4, 20), # we mainly evaluate parameter C for LR
     'classifier__solver' : ['liblinear']}]

clf_cv = GridSearchCV(pipe, param_grid=param_grid, cv=5, verbose=True, n_jobs=-1)
X_train, X_test, y_train, y_test = train_test_split(X_train_scaled, y_train_data, random_state=1)
clf_cv = clf_cv.fit(X_train, y_train)

print(clf_cv.best_params_)
clf_log_best = LogisticRegression(C=clf_cv.best_params_['classifier__C'],
                                  random_state=1,
                                  solver=clf_cv.best_params_['classifier__solver'])
auc_score_log_best, clf_log_best = classifier_assessment(X_train_scaled, y_train_data, clf=clf_log_best)

print("Logistic Regression AUC score (optmimized): ", auc_score_log_best)
importance_dict_lr = feat_importance_eval(clf_log_best)
plot_feature_importance(importance_dict_lr)

## Naive Bayes

clf_nb = GaussianNB(priors=None)
auc_score_nb, clf_nb = classifier_assessment(X_train_scaled, y_train_data, clf=clf_nb)
print("Naive Bayes AUC score: ", auc_score_nb)



## Random Forest
clf_rf = RandomForestClassifier(max_depth=11, random_state=0)
auc_score_rf, clf_rf = classifier_assessment(X_train_scaled, y_train_data, clf=clf_rf)
print("Random Forest AUC score(max depth 11): ", auc_score_rf)
clf_rf_2 = RandomForestClassifier(max_depth=20, random_state=0)
auc_score_rf_2, clf_rf_2 = classifier_assessment(X_train_scaled, y_train_data, clf=clf_rf_2)
print("Random Forest AUC score(max depth 20): ", auc_score_rf_2)
#increasing the depth makes leads to lower AUC score

# Feature Importance
importance_dict_rf = feat_importance_eval(clf_rf)
plot_feature_importance(importance_dict_rf)



## SVM
clf_svc = make_pipeline(StandardScaler(), SVC(gamma='auto', probability=True))
auc_score_svc, clf_svc = classifier_assessment(X_train_scaled, y_train_data, clf=clf_svc)
print("SVM AUC score: ", auc_score_svc)



##XGBoost
xgb_cl = xgb.XGBClassifier(eval_metric='mlogloss')
auc_score_xgb, clf_xgb = classifier_assessment(X_train_scaled, y_train_data, clf=xgb_cl)
print("XGBoost score: ", auc_score_xgb)

param_grid = {
    "max_depth": [1, 3, 5],
    "learning_rate": [0.1, 0.05, 0.01],
    "gamma": [0, 0.25, 1, 10, 20],
    "reg_lambda": [0, 1, 10, 30, 50],
    "scale_pos_weight": [1, 3, 5],
    "subsample": [0.8],
    "colsample_bytree": [0.5],
}

start_time = time.process_time() # evaluate cpu process time in seconds

xgb_cl = xgb.XGBClassifier(eval_metric='mlogloss')
grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=3, scoring="accuracy")
grid_cv_sm = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=3, scoring="accuracy")
_ = grid_cv.fit(X_train_scaled, y_train_data)
_ = grid_cv_sm.fit(X_combine, y_combine)


end_time = time.process_time()

print('Cross validation took time:', end_time-start_time)

best_params = grid_cv.best_params_
print(best_params)

best_xgb = xgb.XGBClassifier(colsample_bytree=best_params['colsample_bytree'],
                             gamma=best_params['gamma'],
                             learning_rate=best_params['learning_rate'],
                             max_depth=best_params['max_depth'],
                             reg_lambda=best_params['reg_lambda'],
                             scale_pos_weight=best_params['scale_pos_weight'],
                             subsample=best_params['subsample'],
                             eval_metric='mlogloss',)

best_params_sm = grid_cv_sm.best_params_
print(best_params_sm)

best_xgb_sm = xgb.XGBClassifier(colsample_bytree=best_params_sm['colsample_bytree'],
                                gamma=best_params_sm['gamma'],
                                learning_rate=best_params_sm['learning_rate'],
                                max_depth=best_params_sm['max_depth'],
                                reg_lambda=best_params_sm['reg_lambda'],
                                scale_pos_weight=best_params_sm['scale_pos_weight'],
                                subsample=best_params_sm['subsample'],
                                eval_metric='mlogloss',)



auc_score_xgb_opt, clf_xgb_best = classifier_assessment(X_train_scaled, y_train_data, clf=best_xgb)
print("XGBoost score (optimized): ", auc_score_xgb_opt)


auc_score_xgb_opt_sm, clf_xgb_best_sm = classifier_assessment(X_combine, y_combine, clf=best_xgb_sm)
print("XGBoost score (using SMOTEEN and optimized): ", auc_score_xgb_opt_sm)

importance_dict_xgb_best = feat_importance_eval(clf_xgb_best)
plot_feature_importance(importance_dict_xgb_best)

importance_dict_xgb_best_sm = feat_importance_eval(clf_xgb_best_sm)
plot_feature_importance(importance_dict_xgb_best_sm)

#make predictions using the clf_xgb_best_sm
y_pred =  clf_xgb_best_sm.predict_proba(X_test_scaled)[:,1]
rec = pd.Series (y_pred, index = X_test_data.index)
print(rec)
print(len(rec))
rec_ord = rec.sort_values()
print(rec_ord)


def engagement_model():
    from sklearn.model_selection import train_test_split
    from sklearn.linear_model import LogisticRegression
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import precision_recall_curve, roc_curve, roc_auc_score
    from sklearn.model_selection import cross_validate, KFold

    # read data and setting video ID as the index
    train_data = pd.read_csv(r'C:\Users\raque\Documents\Applied ML in Python - MU\JupiterNotebookFiles\Files\home\jovyan\work\assignments\course3_assignment4\assets\train.csv').set_index('id')
    features_names = ['title_word_count', 'document_entropy', 'freshness', 'easiness', 'fraction_stopword_presence', 'normalization_rate', 'speaker_speed', 'silent_period_rate']
    X_train_data = train_data[features_names]
    y_train_data = train_data['engagement']

    X_test_data = pd.read_csv(r'C:\Users\raque\Documents\Applied ML in Python - MU\JupiterNotebookFiles\Files\home\jovyan\work\assignments\course3_assignment4\assets\test.csv').set_index('id')

    # scale features using the minmax scaler
    scaling = MinMaxScaler()
    X_train_scaled = scaling.fit_transform(X_train_data)
    X_test_scaled = scaling.transform(X_test_data)

    param_grid = {
        "max_depth": [1, 3, 5],
        "learning_rate": [0.1, 0.05, 0.01],
        "gamma": [0, 0.25, 1, 10, 20],
        "reg_lambda": [0, 1, 10, 30, 50],
        "scale_pos_weight": [1, 3, 5],
        "subsample": [0.8],
        "colsample_bytree": [0.5],
    }

    xgb_cl = xgb.XGBClassifier(eval_metric='mlogloss')
    grid_cv = GridSearchCV(xgb_cl, param_grid, n_jobs=-1, cv=3, scoring="accuracy")
    _ = grid_cv.fit(X_train_scaled, y_train_data)

    best_params = grid_cv.best_params_
    print(best_params)
    best_xgb = xgb.XGBClassifier(colsample_bytree=best_params['colsample_bytree'],
                                 gamma=best_params['gamma'],
                                 learning_rate=best_params['learning_rate'],
                                 max_depth=best_params['max_depth'],
                                 reg_lambda=best_params['reg_lambda'],
                                 scale_pos_weight=best_params['scale_pos_weight'],
                                 subsample=best_params['subsample'],
                                 eval_metric='mlogloss', )

    auc_score_xgb_opt, clf_xgb_best = classifier_assessment(X_train_scaled, y_train_data, clf=best_xgb)

    # make predictions using the clf_xgb_best
    y_pred = clf_xgb_best.predict_proba(X_test_scaled)[:, 1]
    rec = pd.Series(y_pred, index=X_test_data.index)
    print(rec)
    print(len(rec))
    rec_ord = rec.sort_values()
    print(rec_ord)

    return rec



stu_ans = engagement_model()
print(stu_ans)
assert isinstance(stu_ans, pd.Series), "Your function should return a pd.Series. "
assert len(stu_ans) == 2309, "Your series is of incorrect length: expected 2309 "
assert np.issubdtype(stu_ans.index.dtype, np.integer), "Your answer pd.Series should have an index of integer type representing video id."