import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


np.random.seed(0)
n = 15
x = np.linspace(0,10,n) + np.random.randn(n)/5
y = np.sin(x)+x/6 + np.random.randn(n)/10


X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=0)

def intro():


    plt.figure()
    plt.scatter(X_train, y_train, label='training data')
    plt.scatter(X_test, y_test, label='test data')
    plt.legend(loc=4);
    return plt.show()

#print(intro())


def answer_one():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures

    degree_predictions = np.zeros(shape=(4, 100))

    #Generate a list of the polynomial we want
    degrees_list = [1, 3, 6, 9]

    to_predict = np.linspace(0, 10, 100)

    for i, ele in enumerate(degrees_list):

        poly = PolynomialFeatures(degree=ele)
        X_poly = poly.fit_transform(X_train.reshape(11, 1))
        linreg_to_poly = LinearRegression().fit(X_poly, y_train)
        pred_values = linreg_to_poly.predict(poly.fit_transform(to_predict.reshape(100, 1)))
        degree_predictions[i] = pred_values

    return degree_predictions


# feel free to use the function plot_one() to replicate the figure
# from the prompt once you have completed question one
def plot_one(degree_predictions):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10,5))
    plt.plot(X_train, y_train, 'o', label='training data', markersize=10)
    plt.plot(X_test, y_test, 'o', label='test data', markersize=10)
    for i,degree in enumerate([1,3,6,9]):
        plt.plot(np.linspace(0,10,100), degree_predictions[i], alpha=0.8, lw=2, label='degree={}'.format(degree))
    plt.ylim(-1,2.5)
    plt.legend(loc=4)
    return plt.show()

#print(plot_one(answer_one()))



def answer_two():
    from sklearn.linear_model import LinearRegression
    from sklearn.preprocessing import PolynomialFeatures


    r2_train = np.zeros(shape=(10,))
    r2_test = np.zeros(shape=(10,))

    #Generate a list of the polynomial we want
    degrees_list = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]


    for i, ele in enumerate(degrees_list):

        poly = PolynomialFeatures(degree=ele)


        #rsquared training
        X_poly = poly.fit_transform(X_train.reshape(11, 1))
        linreg_to_poly = LinearRegression().fit(X_poly, y_train)
        r2_train[i] = linreg_to_poly.score(X_poly, y_train)

        #rsquared test (do not train)

        X_test_poly = poly.fit_transform(X_test.reshape(4, 1))
        r2_test[i] = linreg_to_poly.score(X_test_poly, y_test)

    return (r2_train, r2_test)


def answer_three():

    r2_scores = answer_two()
    df_scores = pd.DataFrame({'training scores': r2_scores[0], 'test scores': r2_scores[1]})
    print(df_scores)
    df_scores['Diff']= df_scores['training scores'] - df_scores['test scores']
    print(df_scores)


    df_scores_gen = df_scores.sort_values(by=['Diff'])
    print(df_scores_gen)
    Good_Generalization = df_scores_gen.index[0]



    df_scores_overfitting = df_scores.sort_values(by=['Diff'], ascending = False)
    overfitting = df_scores_overfitting.index[0]


    df_scores_underfitting = df_scores.sort_values(by=['training scores'])
    underfitting = df_scores_underfitting.index[0]

    return (underfitting, overfitting, Good_Generalization)

def answer_four():
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import Lasso, LinearRegression
    from sklearn.metrics import r2_score

    #polynomial features of degree 12
    poly = PolynomialFeatures(degree=12)

    #transform for the polynomial
    X_train_poly = poly.fit_transform(X_train.reshape(11, 1))
    X_test_poly = poly.fit_transform(X_test.reshape(4, 1))

    #non-regularized LinearRegression model
    linreg = LinearRegression().fit(X_train_poly,y_train)
    LinearRegression_R2_test_score = linreg.score(X_test_poly, y_test)

    #Lasso Regression model (with parameters alpha=0.01, max_iter=10000, tol=0.1)
    linlasso = Lasso(alpha=0.01, max_iter = 10000, tol=0.1).fit(X_train_poly, y_train)
    Lasso_R2_test_score = linlasso.score(X_test_poly, y_test)

    return (LinearRegression_R2_test_score, Lasso_R2_test_score)



### Classification
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split


mush_df = pd.read_csv(r'C:\Users\raque\Documents\Applied ML in Python - MU\JupiterNotebookFiles\Files\home\jovyan\work\resources\course3\assets\mushrooms.csv')
mush_df2 = pd.get_dummies(mush_df)


X_mush = mush_df2.iloc[:,2:]
y_mush = mush_df2.iloc[:,1]


X_train2, X_test2, y_train2, y_test2 = train_test_split(X_mush, y_mush, random_state=0)

X_subset = X_test2
y_subset = y_test2

def answer_five():
    from sklearn.tree import DecisionTreeClassifier

    clf = DecisionTreeClassifier(random_state=0).fit(X_train2, y_train2)
    df = pd.DataFrame({'features':X_train2.columns.values, 'feature importance': clf.feature_importances_})
    df2 = df.sort_values(['feature importance'], ascending = False)
    #select 5 top rows
    top_features = df2['features'].head(5).tolist()
    return top_features


def answer_six():
    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve

    #create SVC w/ default parameters
    svc_def = SVC(kernel='rfb',C=1, random_state=0)

    #explore the effect of gamma
    gamma = np.logspace(-4,1,6)

    #scores using validation_curve
    training_scores, test_scores = validation_curve(SVC(), X_mush, y_mush, param_name = 'gamma', param_range = gamma, cv=3)
    print(X_subset,y_subset)
    print(training_scores)
    scores = (training_scores.mean(axis = 1 ), test_scores.mean(axis=1))

    return scores


def answer_seven():
    plt.figure()

    from sklearn.svm import SVC
    from sklearn.model_selection import validation_curve
    # explore the effect of gamma
    gamma = np.logspace(-4, 1, 6)
    # scores using validation_curve
    training_scores, test_scores = validation_curve(SVC(), X_mush, y_mush, param_name='gamma', param_range=gamma, cv=3)


    train_scores_mean = np.mean(training_scores,axis=1)
    train_scores_std = np.std(training_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.title('Validation Curve with SVM')
    plt.xlabel('$\gamma$ (gamma)')
    plt.ylabel('Score')
    plt.ylim(0.0, 1.1)
    lw = 2

    plt.semilogx(gamma, train_scores_mean, label='Training score',
                 color='darkorange', lw=lw)

    plt.fill_between(gamma, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color='darkorange', lw=lw)

    plt.semilogx(gamma, test_scores_mean, label='Cross-validation score',
                 color='navy', lw=lw)

    plt.fill_between(gamma, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color='navy', lw=lw)

    plt.legend(loc='best')
    return plt.show()

print (answer_seven())