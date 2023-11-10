import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

fraud_data = pd.read_csv(r'C:\Users\raque\Documents\Applied ML in Python - MU\JupiterNotebookFiles\Files\home\jovyan\work\assignments\course3_assignment3\assets\fraud_data.csv')

class_counts = np.bincount(fraud_data.Class)


X = fraud_data.iloc[:,:-1]
y = fraud_data.iloc[:,-1]

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)


def answer_one():
    #What percentage of the observations in the dataset are instances of fraud?
    total = fraud_data.shape[0]

    counts_group = fraud_data.groupby("Class")["Class"].count()
    pos = counts_group.iloc[1]
    neg = counts_group.iloc[0]

    fraud_perc = (pos/total)

    return fraud_perc


def answer_two():
    from sklearn.dummy import DummyClassifier
    from sklearn.metrics import recall_score
    #dummy classifier
    #Negative class (0) is most frequent
    dummy_majority = DummyClassifier(strategy='most_frequent').fit(X_train, y_train)
    #Therefore the dummy 'most_frequent' classifier always predicts class 0
    y_dummy_predictions = dummy_majority.predict(X_test)
    accuracy_score_r = dummy_majority.score(X_test, y_dummy_predictions)
    recall_score_r = recall_score(y_test, y_dummy_predictions)

    return (accuracy_score_r, recall_score_r)


def answer_three():
    from sklearn.metrics import recall_score, precision_score
    from sklearn.svm import SVC

    svc_class = SVC().fit(X_train, y_train)
    accuracy = svc_class.score(X_test, y_test)

    svc_predicted = svc_class.predict(X_test)

    recall = recall_score(y_test, svc_predicted)
    precision = precision_score(y_test, svc_predicted)
    return (accuracy, recall, precision)





def answer_four():
    from sklearn.metrics import confusion_matrix
    # Accuracy of Support Vector Machine classifier
    from sklearn.svm import SVC

    m = SVC(C=1e9, gamma = 1e-07)
    m_trained = m.fit(X_train, y_train)
    y_scores = m_trained.decision_function(X_test) > -220

    confusion = confusion_matrix(y_test, y_scores)
    return confusion

def answer_five():
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import precision_recall_curve, roc_curve


    lr = LogisticRegression(solver='liblinear').fit(X_train, y_train)

    y_scores_lr = lr.decision_function(X_test)

    precision, recall, thresholds = precision_recall_curve(y_test, y_scores_lr)

    fpr_lr, tpr_lr, _ = roc_curve(y_test, y_scores_lr)

    rec = float(recall[np.where(precision == 0.75)])


    tpr = float(tpr_lr[np.where(fpr_lr >= 0.16)][0])




    return(rec, tpr)




def answer_six():
    from sklearn.model_selection import GridSearchCV
    from sklearn.linear_model import LogisticRegression

    lr = LogisticRegression(solver='liblinear')
    grid_values = {"penalty": ["l1", "l2"], "C": [0.01, 0.1, 1, 10]}

    lr_grids = GridSearchCV(lr, param_grid=grid_values, scoring='recall')
    lr_grids.fit(X_train, y_train)
    ar_mean_test_score = np.array(lr_grids.cv_results_["mean_test_score"])

    result_array = ar_mean_test_score.reshape(4, 2)

    return result_array

# Use the following function to help visualize results from the grid search
def GridSearch_Heatmap(scores):

    import seaborn as sns
    import matplotlib.pyplot as plt
    plt.figure()
    sns.heatmap(scores.reshape(4,2), xticklabels=['l1','l2'], yticklabels=[0.01, 0.1, 1, 10])
    plt.yticks(rotation=0)
    plt.show()

print(GridSearch_Heatmap(answer_six()))


print(answer_five())
print(answer_six())