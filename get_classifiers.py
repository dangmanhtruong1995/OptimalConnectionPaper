from sklearn import svm
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
# Multiclass classifiers
from sklearn.multiclass import OneVsRestClassifier
# from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB, ComplementNB
from sklearn.naive_bayes import BernoulliNB, GaussianNB, MultinomialNB
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
# from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier, HistGradientBoostingClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF, DotProduct, ConstantKernel as C, Matern, RationalQuadratic, \
    WhiteKernel, ExpSineSquared, Product, Sum
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier, NearestCentroid
from sklearn.svm import LinearSVC, SVC
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV, RidgeClassifier, RidgeClassifierCV, \
    Perceptron, PassiveAggressiveClassifier, SGDClassifier
from sklearn.neural_network import MLPClassifier, BernoulliRBM
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
import xgboost as xgb
from xgboost import XGBClassifier
import numpy as np
import pdb
from pdb import set_trace
from warnings import filterwarnings
from pprint import pprint
#xgb.set_config(eval_metric='merror')

filterwarnings('ignore')


class GaussianProcessClassifierRBF(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.clf = GaussianProcessClassifier(kernel=1.0 * RBF(np.ones((X.shape[1])))).fit(X, y)
#        return self.clf

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class GaussianProcessClassifierRBFMultipliedByRationalQuadratic(object):
    def __init__(self):
        pass

    def fit(self, X, y):
        self.clf = GaussianProcessClassifier(Product(1.0 * RBF(np.ones((X.shape[1])) \
                                                               ), RationalQuadratic())).fit(X, y)
        return self.clf


class XGBClassifierWrapper(object):
    def __init__(self, n_estimators):
        self.n_estimators = n_estimators

    def fit(self, X, y):
        y1 = y-1 # [1, 2, 3] -> [0, 1, 2]
        num_of_classes = np.size(np.unique(y1))
        if num_of_classes == 2:
            self.clf = XGBClassifier(
                n_estimators=self.n_estimators,
                object='binary:logistic',
                verbosity=0,
                silent=True,
                eval_metric='merror').fit(X, y1)
        else:
            self.clf = XGBClassifier(
                n_estimators=self.n_estimators,
                object='multi:softmax',
                verbosity=0,
                silent=True,
                eval_metric='merror').fit(X, y1)
        # self.clf.eval_metric = 'merror' # Update from XGboost 1.3.0

    def predict(self, X):
        return self.clf.predict(X)+1 # [0, 1, 2] -> [1, 2, 3]

    def predict_proba(self, X):
        return self.clf.predict_proba(X)


class KNNWrapper(object):
    def __init__(self, percentage=0.1, weighting='uniform'):
        self.percentage = percentage
        self.weighting = weighting

    def fit(self, X, y):
        num_of_neighbors = int(X.shape[0] * self.percentage)
        if num_of_neighbors == 0:
            num_of_neighbors = 1
        return KNeighborsClassifier(n_neighbors=int(X.shape[0] * self.percentage), weights=self.weighting).fit(X, y)


def get_classifiers_2():
    clf_list = []
    clf_name_list = []

    # Add Decision tree
    criterion_list = ['gini', 'entropy']
    splitter_list = ['best', 'random']
#    min_impurity_decrease_list = [0.0, 0.2, 0.4, 0.6, 0.8]
    for criterion in criterion_list:
#        for splitter in splitter_list:
#            for min_impurity_decrease in min_impurity_decrease_list:
#                clf_name = 'decision_tree-%s-%s-%.1f' % (criterion, splitter, min_impurity_decrease)
#                clf_list.append(DecisionTreeClassifier(
#                    criterion = criterion,
#                    splitter = splitter,
#                    min_impurity_decrease=min_impurity_decrease,
#                            ))

        for splitter in splitter_list:
            clf_name = 'decision_tree-%s-%s' % (criterion, splitter)
            clf_list.append(DecisionTreeClassifier(
                criterion=criterion,
                splitter=splitter,
            ))
            clf_name_list.append(clf_name)

    return clf_list, clf_name_list

def get_classifiers():
    clf_list = []
    clf_name_list = []

    # Add Decision tree
    criterion_list = ['gini', 'entropy']
    splitter_list = ['best', 'random']
#    min_impurity_decrease_list = [0.0, 0.2, 0.4, 0.6, 0.8]
    for criterion in criterion_list:
#        for splitter in splitter_list:
#            for min_impurity_decrease in min_impurity_decrease_list:
#                clf_name = 'decision_tree-%s-%s-%.1f' % (criterion, splitter, min_impurity_decrease)
#                clf_list.append(DecisionTreeClassifier(
#                    criterion = criterion,
#                    splitter = splitter,
#                    min_impurity_decrease=min_impurity_decrease,
#                            ))

        for splitter in splitter_list:
            clf_name = 'decision_tree-%s-%s' % (criterion, splitter)
            clf_list.append(DecisionTreeClassifier(
                criterion=criterion,
                splitter=splitter,
            ))
            clf_name_list.append(clf_name)

    # Add kNN
    n_neighbors_list = [1, 3, 5, 7, 9]
    for n_neighbors in n_neighbors_list:
        clf_list.append(KNeighborsClassifier(n_neighbors=n_neighbors))
        clf_name_list.append('knn_%d' % (n_neighbors))

    # Add SVM
#    kernel_list = ['linear', 'rbf', 'poly', 'sigmoid']
#    for kernel in kernel_list:
#        clf_list.append(OneVsRestClassifier(SVC(kernel=kernel, probability=True, gamma='auto')))
#        clf_name_list.append('svm_%s' % (kernel))

    # Add Random Forest
    clf_list.append(RandomForestClassifier(n_estimators = 200))
    clf_name_list.append('random_forest')

    # Add Bagging
    clf_list.append(BaggingClassifier(n_estimators = 200))
    clf_name_list.append('bagging')

    # Add XGBoost
    clf_list.append(XGBClassifierWrapper(n_estimators = 200))
    clf_name_list.append('xgboost')

    # Add MLP
    hidden_layer_size_list = [20, 40, 60, 80, 100]
    learning_rate_list = [0.3, 0.6]
    for hidden_layer_size in hidden_layer_size_list:
        for learning_rate in learning_rate_list:
            clf_name = 'mlp-hidden_layer_%.3f-learning_rate_%.3f' % (hidden_layer_size, learning_rate)
            clf_list.append(MLPClassifier(
                hidden_layer_sizes = (hidden_layer_size,),
                learning_rate = 'constant',
                learning_rate_init = learning_rate))
            clf_name_list.append(clf_name)

    # Add Gaussian NB
    clf_list.append(GaussianNB())
    clf_name_list.append('gaussian_nb')

    # Add Adaboost
    num_of_estimators_list = [200]
    for num_of_estimators in num_of_estimators_list:
        clf_name = 'adaboost-%d' % (num_of_estimators)
        clf_list.append(AdaBoostClassifier(n_estimators=num_of_estimators))
        clf_name_list.append(clf_name)

    # Add Logistic regression
    C_list = [0.001, 0.01, 0.1, 1, 10, 100]
    for C in C_list:
        clf_name = 'logistic_regression-%.3f' % (C)
        clf_list.append(LogisticRegression(solver = 'newton-cg', C = C))
        clf_name_list.append(clf_name)

    # Add Gaussian Process
#    clf_list.append(GaussianProcessClassifierRBF())
#    clf_name_list.append('gaussian_process_rbf')
#    clf_list.append(GaussianProcessClassifier(kernel=Matern()))
#    clf_name_list.append('gaussian_process_matern')
#    clf_list.append(GaussianProcessClassifier(kernel=RationalQuadratic()))
#    clf_name_list.append('gaussian_process_rational_quadratic')


    return clf_list, clf_name_list

def main():
    clfs_list, clfs_name_list = get_classifiers()
    set_trace()
    
if __name__ == "__main__":
    main()