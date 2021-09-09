import pandas as pd
from numpy import mean
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn import svm
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier
from sklearn import tree
import random

knn = KNeighborsClassifier()
svm = svm.SVC()
lr = LogisticRegression()
nb = GaussianNB()
mlp = MLPClassifier()
dt_gini = tree.DecisionTreeClassifier(criterion='gini')
dt_entropy = tree.DecisionTreeClassifier(criterion='entropy')
acc_knn = []
acc_svm = []
acc_lr = []
acc_nb = []
acc_mlp = []
acc_dt_gini = []
acc_dt_entropy = []

dir_name = "../datasets/csv/"
GSES = ['GSE14728','GSE42408', 'GSE46205', 'GSE76613', 'GSE145709']
resultspath = '../JILU/'


def main():
    with open(resultspath + 'otherClassifier.txt', 'w') as f:
        f.write('DATASET\t\tKNN\t\tSVM\t\tLR\t\tnb\t\tmlp\t\tdt_gini\t\tdt_entropy\n')
        for GSE in GSES:
            dataset = pd.read_csv(dir_name + GSE + '.csv')
            X = dataset.iloc[:, :-1].values
            y = dataset.iloc[:, -1].values
            for i in range(30):
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3,
                                                                    random_state=random.randint(0, 1000))

                knn.fit(X_train, y_train)
                acc_knn.append(knn.score(X_test, y_test))

                svm.fit(X_train, y_train)
                acc_svm.append(svm.score(X_test, y_test))

                lr.fit(X_train, y_train)
                acc_lr.append(lr.score(X_test, y_test))

                nb.fit(X_train, y_train)
                acc_nb.append(nb.score(X_test, y_test))

                mlp.fit(X_train, y_train)
                acc_mlp.append(mlp.score(X_test, y_test))

                dt_gini.fit(X_train, y_train)
                acc_dt_gini.append(dt_gini.score(X_test, y_test))

                dt_entropy.fit(X_train, y_train)
                acc_dt_entropy.append(dt_entropy.score(X_test, y_test))

            f.write(GSE + '\t' + str(round(mean(acc_knn) * 100, 3)) + '\t'
                    + str(round(mean(acc_svm) * 100, 3)) + '\t'
                    + str(round(mean(acc_lr) * 100, 3)) + '\t'
                    + str(round(mean(acc_nb) * 100, 3)) + '\t'
                    + str(round(mean(acc_mlp) * 100, 3)) + '\t'
                    + str(round(mean(acc_dt_gini) * 100, 3)) + '\t'
                    + str(round(mean(acc_dt_entropy) * 100, 3)) + '\n')


if __name__ == '__main__':
    main()
