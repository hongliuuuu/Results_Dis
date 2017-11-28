from sklearn.kernel_approximation import (RBFSampler,Nystroem)
from sklearn.ensemble import RandomForestClassifier
import pandas
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel,laplacian_kernel,chi2_kernel,linear_kernel,polynomial_kernel,cosine_similarity
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import xlrd
import xlrd
import numpy as np
import pandas
import random
import time
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.externals import joblib
from sklearn.pipeline import make_pipeline
from skrebate import ReliefF
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import roc_auc_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing
from collections import Counter
import re
from math import floor
from joblib import Parallel, delayed
np.set_printoptions(threshold=np.nan)

def floored_percentage(val, digits):
    val *= 10 ** (digits + 2)
    return '{1:.{0}f}\%\pm '.format(digits, floor(val) / 10 ** digits)
def splitdata(X,Y,ratio,seed):
    '''This function is to split the data into train and test data randomly and preserve the pos/neg ratio'''
    n_samples = X.shape[0]
    y = Y.astype(int)
    y_bin = np.bincount(y)
    classes = np.nonzero(y_bin)[0]
    #fint the indices for each class
    indices = []
    print()
    for i in classes:
        indice = []
        for j in range(n_samples):
            if y[j] == i:
                indice.append(j)
        #print(len(indice))
        indices.append(indice)
    train_indices = []
    for i in indices:
        k = int(len(i)*ratio)
        train_indices += (random.Random(seed).sample(i,k=k))
    #find the unused indices
    s = np.bincount(train_indices,minlength=n_samples)
    mask = s==0
    test_indices = np.arange(n_samples)[mask]
    return train_indices,test_indices
"""
def rf_dis(n_trees, X,Y,train_indices,test_indices,seed,wei):
    clf = RandomForestClassifier(n_estimators=n_trees,
                                 random_state=seed, oob_score=False, n_jobs=1)
    clf = clf.fit(X[train_indices], Y[train_indices])
    pred = clf.predict(X[test_indices])
    prediction = clf.predict(X)
    prob = clf.predict_proba(X[test_indices])
    weight =0#clf.oob_score_ #clf.score(X[test_indices], Y[test_indices])
    #print(clf.score(X[train_indices], Y[train_indices]))
    #print(1 - clf.oob_score_)
    n_samples = X.shape[0]
    trees = clf.estimators_
    dis = np.zeros((n_samples,n_samples))
    for i in range(n_samples):
        dis[i][i] = 1
    res = clf.apply(X)
    www = wei
    pre = np.zeros((n_trees, n_samples))
    pre = pre.transpose()
    for i in range(n_samples):
        for j in range(i+1,n_samples):
            a = np.ravel(res[i])
            b = np.ravel(res[j])
            c = np.ravel(pre[i])
            d = np.ravel(pre[j])

            score = 0
            for k in range(n_trees):
                if a[k] == b[k]:
                    s1=1
                else:
                    s1 = 0
                if c[k] == d[k]:
                    s2=1
                else:
                    s2 = 0
                s = s1*www + s2*(1-www)
                score = score + s
            dis[i][j]  =dis[j][i] = score/n_trees
    X_features1 = np.transpose(dis)
    X_features2 = X_features1[train_indices]
    X_features3 = np.transpose(X_features2)

    return X_features3[train_indices],X_features3[test_indices],weight,pred,prob,clf
"""
def rf_dis(n_trees, X,Y,train_indices,test_indices,seed, wei):
    clf = RandomForestClassifier(n_estimators=500,
                                 random_state=seed, oob_score=True, n_jobs=1)
    clf = clf.fit(X[train_indices], Y[train_indices])
    pred = clf.predict(X[test_indices])
    prob = clf.predict_proba(X[test_indices])
    weight =clf.oob_score_ #clf.score(X[test_indices], Y[test_indices])
    #print(1 - clf.oob_score_)
    n_samples = X.shape[0]
    dis = np.zeros((n_samples,n_samples))
    trees = clf.estimators_
    www = wei
    for i in range(n_samples):
        dis[i][i] = 1
    for k in range(len(trees)):
        pa = trees[k].decision_path(X)
        for i in range(n_samples):
            for j in range(i+1,n_samples):
                a = pa[i]
                a = a.toarray()
                a = np.ravel(a)

                b = pa[j]
                b = b.toarray()
                b = np.ravel(b)
                score = a == b
                d = score.sum()-len(a)
                dis[i][j] = dis[j][i] = dis[i][j]+np.exp(www*d)
    dis = dis/n_trees
    X_features1 = np.transpose(dis)
    X_features2 = X_features1[train_indices]
    X_features3 = np.transpose(X_features2)
    return X_features3[train_indices],X_features3[test_indices],weight,pred,prob,clf
def gama_patatune(train_x,train_y,c):
    tuned_parameters = [
        {'kernel': ['rbf'], 'gamma': [0.0625, 0.125,0.25, 0.5, 1, 2, 5 ,7, 10, 12 ,15 ,17 ,20] }]
    clf = GridSearchCV(SVC(C=c), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    return clf.best_params_['gamma']


def relf(n_neb, n_feat, trainx, trainy,testx):
    fs = ReliefF(n_features_to_select=n_feat, n_neighbors=n_neb,discrete_threshold=10, n_jobs=1)
    fs.fit(trainx, trainy)
    ind = fs.transform(trainx)
    return ind

def lsvm_rfe(c,n_feat,trainX,trainy, testX):
    svc = SVC(kernel="linear", C=c)
    rfe = RFE(estimator=svc, n_features_to_select=n_feat, step=1)
    rfe.fit(trainX, trainy)
    train_X = rfe.transform(trainX)
    test_X = rfe.transform(testX)
    return train_X,test_X
def RF(n_trees,  seed, train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier(n_estimators=n_trees,
                                  random_state = seed, oob_score=True)
    clf = clf.fit(train_x,train_y)
    oob_error = 1 - clf.oob_score_
    test_error = clf.score(test_x,test_y)
    test_auc = clf.predict_proba(test_x)
    #filename = './tmp1/RF_%d_.pkl'%seed
    #_ = joblib.dump(clf, filename, compress=9)
    return test_error, test_auc
def selected_f(n_features):
    if n_features>1000:
        n = 25
    elif n_features>100:
        n = int(n_features*0.03)
    elif n_features >75:
        n = int(n_features * 0.1)
    else :
        n = int(n_features * 0.4)
    return n
def nLsvm_patatune(train_x,train_y,test_x, test_y):
    tuned_parameters = [
        {'kernel': ['precomputed'], 'C': [0.01, 0.1, 1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(C=1), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    #print(clf.score(test_x,test_y))
    return clf.best_params_['C']
def Lsvm_patatune(train_x,train_y):
    tuned_parameters = [
        {'kernel': ['linear'], 'C': [0.01,0.1, 1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(C=1, probability=True), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    return clf.best_params_['C']

def weightedComb(Y, W):
    y = Y.astype(int)
    y_bin = np.bincount(y)
    classes = np.nonzero(y_bin)[0]
    # fint the indices for each class
    indices = []

    for i in classes:
        pro = 0
        indice = []
        for j in range(len(y)):
            if y[j] == i:
                indice.append(j)
                pro = pro + W[j]
        indices.append(pro)
    ind = (list(indices)).index(max(indices))
    return classes[ind]
url = 'text_pr_1.csv'
dataframe = pandas.read_csv(url, header=None)
array = dataframe.values
X = array
Y = pandas.read_csv('label_progression.csv', header=None)
Y = Y.values
Y = np.ravel(Y)
print(Y.shape)

for i in range(4):
    url = 'text_pr_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url, header=None)
    array = dataframe.values
    X1 = array
    print(X1.shape)
    X = np.concatenate((X, X1), axis=1)
Progression = X
Progression1 = X[:, 0:1680]
Progression2 = X[:, 1680:3360]
Progression3 = X[:, 3360:5040]
Progression4 = X[:, 5040:6720]
Progression5 = X[:, 6720:6745]
ProgressionY = Y
url = 'text_lg_1.csv'
dataframe = pandas.read_csv(url, header=None)
array = dataframe.values
X = array
Y = pandas.read_csv('label_lowGrade.csv', header=None)
Y = Y.values
Y = np.ravel(Y)
print(Y.shape)

for i in range(4):
    url = 'text_lg_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url, header=None)
    array = dataframe.values
    X1 = array
    print(X1.shape)
    X = np.concatenate((X, X1), axis=1)
lowGrade = X
lowGrade1 = X[:, 0:1680]
lowGrade2 = X[:, 1680:3360]
lowGrade3 = X[:, 3360:5040]
lowGrade4 = X[:, 5040:6720]
lowGrade5 = X[:, 6720:6745]
lowGradeY = Y
url = 'text_nonIDH1_1.csv'
dataframe = pandas.read_csv(url, header=None)
array = dataframe.values
X = array
Y = pandas.read_csv('label_nonIDH1.csv', header=None)
Y = Y.values
Y = np.ravel(Y)
print(Y.shape)

for i in range(4):
    url = 'text_nonIDH1_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url, header=None)
    array = dataframe.values
    X1 = array
    print(X1.shape)
    X = np.concatenate((X, X1), axis=1)
nonIDH=X
nonIDH1 = X[:, 0:1680]
nonIDH2 = X[:, 1680:3360]
nonIDH3 = X[:, 3360:5040]
nonIDH4 = X[:, 5040:6720]
nonIDH5 = X[:, 6720:6745]
nonIDHY = Y
url = 'text_id_1.csv'
dataframe = pandas.read_csv(url, header=None)
array = dataframe.values
X = array
Y = pandas.read_csv('label_IDHCodel.csv', header=None)
Y = Y.values
Y = np.ravel(Y)
print(Y.shape)

for i in range(4):
    url = 'text_id_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url, header=None)
    array = dataframe.values
    X1 = array
    print(X1.shape)
    X = np.concatenate((X, X1), axis=1)
IDHCodel=X
IDHCodel1 = X[:, 0:1680]
IDHCodel2 = X[:, 1680:3360]
IDHCodel3 = X[:, 3360:5040]
IDHCodel4 = X[:, 5040:6720]
IDHCodel5 = X[:, 6720:6745]
IDHCodelY = Y

def mcode(ite):

    R = 0.5
    seed = 1000 + ite
    numberofclass = 2
    for ddd in range(4):
        if ddd ==0:
            X = IDHCodel
            Y = IDHCodelY
            fff = "SPBKNDIDHCodelwww4%f_%f" % (R, ite)
        if ddd ==1:
            X = nonIDH
            Y = nonIDHY
            fff = "SPBKNDnonIDHwww4%f_%f" % (R, ite)
        if ddd ==2:
            X = lowGrade
            Y = lowGradeY
            fff = "SPBKNDLGwww4%f_%f" % (R, ite)
        if ddd ==3:
            X = Progression
            Y = ProgressionY
            fff ="SPBKNDprwww4%f_%f" % (R, ite)
        Xnew1 = X[:, 0:1680]
        Xnew2 = X[:, 1680:3360]
        Xnew3 = X[:, 3360:5040]
        Xnew4 = X[:, 5040:6720]
        Xnew5 = X[:, 6720:6745]
        train_indices, test_indices = splitdata(X=X, Y=Y, ratio=R, seed=seed)

        for ii in range(20):
            fn = fff+"ite%d"%(ii)+".txt"
            testfile = open(fn, 'w')
            ndw = 0.1*(ii+1)
            X_features_train1, X_features_test1, w1, pred1, prob1, RFV1 = rf_dis(n_trees=500, X=Xnew1, Y=Y, train_indices=train_indices,
                                                                           test_indices=test_indices, seed=seed, wei=ndw)
            X_features_train2, X_features_test2, w2, pred2, prob2, RFV2 = rf_dis(n_trees=500, X=Xnew2, Y=Y,
                                                                                 train_indices=train_indices,
                                                                                 test_indices=test_indices, seed=seed, wei=ndw)
            X_features_train3, X_features_test3, w3, pred3, prob3, RFV3 = rf_dis(n_trees=500, X=Xnew3, Y=Y,
                                                                                 train_indices=train_indices,
                                                                                 test_indices=test_indices, seed=seed, wei=ndw)
            X_features_train4, X_features_test4, w4, pred4, prob4, RFV4 = rf_dis(n_trees=500, X=Xnew4, Y=Y,
                                                                                 train_indices=train_indices,
                                                                                 test_indices=test_indices, seed=seed, wei=ndw)
            X_features_train5, X_features_test5, w5, pred5, prob5, RFV5 = rf_dis(n_trees=500, X=Xnew5, Y=Y,
                                                                                 train_indices=train_indices,
                                                                                 test_indices=test_indices, seed=seed, wei=ndw)

            # multi view
            X_features_trainm = (
                                    X_features_train1 + X_features_train2 + X_features_train3 + X_features_train4 + X_features_train5) / 5
            X_features_testm = (
                                   X_features_test1 + X_features_test2 + X_features_test3 + X_features_test4 + X_features_test5) / 5

            mv = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
                X_features_trainm, Y[train_indices])
            R1=(mv.score(X_features_testm, Y[test_indices]))

            # RFSVM
            c = nLsvm_patatune(train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                               test_y=Y[test_indices])

            clf = SVC(C=c, kernel='precomputed')
            clf.fit(X_features_trainm, Y[train_indices])
            R2=(clf.score(X_features_testm, Y[test_indices]))
            """
            # W multi view
            X_features_trainm = (
                                    X_features_train1*W[0] + X_features_train2*W[1] + X_features_train3*W[2] + X_features_train4*W[3] + X_features_train5*W[4]) / 5
            X_features_testm = (
                                   X_features_test1*W[0] + X_features_test2*W[1] + X_features_test3*W[2] + X_features_test4*W[3] + X_features_test5*W[4]) / 5
            mv = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
                X_features_trainm, Y[train_indices])
            R3=(mv.score(X_features_testm, Y[test_indices]))

            # RFSVM
            c = nLsvm_patatune(train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                               test_y=Y[test_indices])

            clf = SVC(C=c, kernel='precomputed')
            clf.fit(X_features_trainm, Y[train_indices])
            R4=(clf.score(X_features_testm, Y[test_indices]))

            # weight multi view
            X_features_trainm = (
                                    X_features_train1 * weight[0] + X_features_train2 * weight[1] + X_features_train3 * weight[
                                        2] + X_features_train4 * weight[3] + X_features_train5 * weight[4]) / 5
            X_features_testm = (
                                   X_features_test1 * weight[0] + X_features_test2 * weight[1] + X_features_test3 * weight[
                                       2] + X_features_test4 * weight[3] + X_features_test5* weight[4]) / 5
            mv = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
                X_features_trainm, Y[train_indices])
            R5=(mv.score(X_features_testm, Y[test_indices]))

            # RFSVM
            c = nLsvm_patatune(train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                               test_y=Y[test_indices])

            clf = SVC(C=c, kernel='precomputed')
            clf.fit(X_features_trainm, Y[train_indices])
            R6=(clf.score(X_features_testm, Y[test_indices]))
            """
            testfile.write(" R1&%s pm%s &" % (floored_percentage(np.mean(R1), 2), floored_percentage(np.std(R1), 2)) + '\n')
            testfile.write(" R2&%s pm%s &" % (floored_percentage(np.mean(R2), 2), floored_percentage(np.std(R2), 2)) + '\n')
            #testfile.write(" R3&%s pm%s &" % (floored_percentage(np.mean(R3), 2), floored_percentage(np.std(R3), 2)) + '\n')
            #testfile.write(" R4&%s pm%s &" % (floored_percentage(np.mean(R4), 2), floored_percentage(np.std(R4), 2)) + '\n')
            #testfile.write(" R5&%s pm%s &" % (floored_percentage(np.mean(R5), 2), floored_percentage(np.std(R5), 2)) + '\n')
            #testfile.write(" R6&%s pm%s &" % (floored_percentage(np.mean(R6), 2), floored_percentage(np.std(R6), 2)) + '\n')
            testfile.close()

if __name__ == '__main__':
    Parallel(n_jobs=10)(delayed(mcode)(ite=i) for i in range(10))

