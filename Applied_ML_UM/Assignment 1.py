import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_breast_cancer

cancer = load_breast_cancer()

#print(cancer.DESCR) # Print the data set description

print(cancer.keys())

def answer_zero():
    return len(cancer['feature_names'])



def answer_one():
    # Your code here
    data = pd.DataFrame(cancer.data, columns=cancer.feature_names)
    data['target'] = cancer.target
    return data


def answer_two():
    cancer_df = answer_one()

    index = ['malignant', 'benign']
    malignant = np.where(cancer_df['target'] == 0.0);
    benign = np.where(cancer_df['target'] == 1.0);
    data_2 = [np.size(malignant), np.size(benign)]

    x = pd.Series(data_2, index=index)

    return x


def answer_three():
    cancer_df = answer_one()
    X = cancer_df[cancer_df.columns[:-1]]
    y = cancer_df['target']

    return X, y


def answer_four():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    X_train_s = X_train.shape
    X_test_s = X_test.shape
    y_train_s = y_train.shape
    y_test_s = y_test.shape

    group_shapes = (X_train_s, X_test_s, y_train_s, y_test_s)
    return group_shapes


def answer_five():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    knn = KNeighborsClassifier(n_neighbors=1)
    knn.fit(X_train, y_train)

    return knn





def answer_six():
    cancer_df = answer_one()
    #gets the mean value for each feature, ignores the target column, and reshapes the data from 1 dimension to 2, necessary for the precict method of KNeighborsClassifier
    means = cancer_df.mean()[:-1].values.reshape(1, -1)
    knn = answer_five()

    return knn.predict(means)




def answer_seven():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn = answer_five()
    test_predict = knn.predict(X_test)
    print(test_predict, test_predict.shape)

    return test_predict



def answer_eight():
    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)
    knn = answer_five()
    #Estimate the accuracy of the classifier on future data, using the test data using knn.score
    return knn.score(X_test, y_test)




def accuracy_plot():
    import matplotlib.pyplot as plt



    X, y = answer_three()
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

    # training and testing accuracies by target value (i.e. malignant, benign)
    mal_X_train = X_train[y_train==0]
    mal_y_train = y_train[y_train==0]
    ben_X_train = X_train[y_train==1]
    ben_y_train = y_train[y_train==1]

    mal_X_test = X_test[y_test==0]
    mal_y_test = y_test[y_test==0]
    ben_X_test = X_test[y_test==1]
    ben_y_test = y_test[y_test==1]

    knn = answer_five()

    scores = [knn.score(mal_X_train, mal_y_train), knn.score(ben_X_train, ben_y_train),
              knn.score(mal_X_test, mal_y_test), knn.score(ben_X_test, ben_y_test)]

    plt.figure()
    #Plot the scores as a bar chart
    bars = plt.bar(np.arange(4), scores, color=['#4c72b0', '#4c72b0', '#55a868', '#55a868'])

    #directly label the score onto the bars
    for bar in bars:
         height = bar.get_height()
         plt.gca().text(bar.get_x() + bar.get_width() / 2, height * .90, '{0:.{1}f}'.format(height, 2),
                     ha='center', color='w', fontsize=11)

    #remove all the ticks (both axes), and tick labels on the Y axis
    plt.tick_params(top='off', bottom='off', left='off', right='off', labelleft='off', labelbottom='on')

    plt.xticks([0, 1, 2, 3], ['Malignant\nTraining', 'Benign\nTraining', 'Malignant\nTest', 'Benign\nTest'], alpha=0.8);
    plt.title('Training and Test Accuracies for Malignant and Benign Cells', alpha=0.8)
    return plt.show()

print(answer_four())
print(type(answer_four()))



#
# # remove the frame of the chart
# for spine in plt.gca().spines.values():
#     spine.set_visible(False)
#

