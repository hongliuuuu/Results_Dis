import xlrd
import numpy as np
import pandas
import random
import time
from sklearn.metrics import accuracy_score
from sklearn import model_selection
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
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
from joblib import Parallel, delayed
from sklearn.ensemble import RandomForestClassifier
import pandas
import numpy as np
import random
from sklearn.svm import SVC
from sklearn.metrics.pairwise import rbf_kernel,laplacian_kernel,chi2_kernel,linear_kernel,polynomial_kernel,cosine_similarity
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import xlrd

def splitdata(X,Y,ratio,seed):
    '''This function is to split the data into train and test data randomly and preserve the pos/neg ratio'''
    n_samples = X.shape[0]
    y = Y.astype(int)
    y_bin = np.bincount(y)
    classes = np.nonzero(y_bin)[0]
    #fint the indices for each class
    indices = []
    for i in classes:
        indice = []
        for j in range(n_samples):
            if y[j] == i:
                indice.append(j)
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

def knn(n_neb, train_x, train_y, test_x, test_y):
    clf =KNeighborsClassifier(n_neighbors=n_neb, n_jobs=1)
    clf.fit(train_x, train_y)
    test_error = clf.score(test_x, test_y)
    test_auc = clf.predict_proba(test_x)
    return test_error, test_auc

def relf(n_neb, n_feat, X, y):
    fs = ReliefF(n_features_to_select=n_feat, n_neighbors=n_neb,discrete_threshold=0, n_jobs=1)
    fs.fit(X, y)
    New_X = fs.transform(X)
    return New_X

def combine_rfs(rf_a, rf_b):
    rf_a.estimators_ += rf_b.estimators_
    rf_a.n_estimators = len(rf_a.estimators_)
    return rf_a

def RF(n_trees,  seed, train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier(n_estimators=n_trees,
                                  random_state = seed, oob_score=True,n_jobs=1)
    clf = clf.fit(train_x,train_y)
    oob_error = 1 - clf.oob_score_
    test_error = clf.score(test_x,test_y)
    test_auc = clf.predict_proba(test_x)
    #filename = './tmp1/RF_%d_.pkl'%seed
    #_ = joblib.dump(clf, filename, compress=9)
    return test_error, test_auc

def LDA(train_x, train_y, test_x, test_y):
    clf = LinearDiscriminantAnalysis()
    clf.fit(train_x,train_y)
    test_error = clf.score(test_x, test_y)
    test_auc = clf.predict_proba(test_x)
    return test_error, test_auc

def Lsvm_patatune(train_x,train_y):
    tuned_parameters = [
        {'kernel': ['linear'], 'C': [0.1, 1, 10, 100, 1000, 10000]}]
    clf = GridSearchCV(SVC(C=1, probability=True), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    return clf.best_params_['C']


def Lsvm(c,train_x, train_y, test_x, test_y):
    clf = SVC(kernel="linear", C=c, probability=True)
    clf.fit(train_x,train_y)
    test_error = clf.score(test_x, test_y)
    test_auc = clf.predict_proba(test_x)
    return test_error,test_auc

def Nsvm(train_x, train_y, test_x, test_y):
    tuned_parameters = [{'kernel': ['rbf'], 'gamma': [0.0625, 0.125,0.25, 0.5, 1, 2, 5 ,7, 10, 12 ,15 ,17 ,20],
                         'C': [0.1, 1, 10, 100, 1000,10000]}
                        ]
    clf = GridSearchCV(SVC(C=1,probability=True), tuned_parameters, cv=5, n_jobs=1
                       )#SVC(probability=True)
    clf.fit(train_x,train_y)
    test_error = clf.score(test_x, test_y)
    test_auc = clf.predict_proba(test_x)

    return test_error,test_auc

def rf_dis(n_trees, X,train_indices,test_indices,seed):
    clf = RandomForestClassifier(n_estimators=500,
                                 random_state=seed, oob_score=True, n_jobs=1)
    clf = clf.fit(X[train_indices], Y[train_indices])
    weight = clf.score(X[test_indices], Y[test_indices])
    #print(1 - clf.oob_score_)
    n_samples = X.shape[0]
    dis = np.zeros((n_samples,n_samples))
    for i in range(n_samples):
        dis[i][i] = 0
    res = clf.apply(X)
    for i in range(n_samples):
        for j in range(i+1,n_samples):
            a = np.ravel(res[i])
            b = np.ravel(res[j])
            score = a == b
            d = float(score.sum())/n_trees
            dis[i][j]  =dis[j][i] = d
    X_features1 = np.transpose(dis)
    X_features2 = X_features1[train_indices]
    X_features3 = np.transpose(X_features2)
    return X_features3[train_indices],X_features3[test_indices],weight





url = 'Cal7_1.csv'
dataframe = pandas.read_csv(url)  # , header=None)
array = dataframe.values
X = array[:, 1:]

for i in range(5):
    url = 'Cal7_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url)  # , header=None)
    array = dataframe.values
    X1 = array[:, 1:]
    X = np.concatenate((X, X1), axis=1)
Y = pandas.read_csv('Cal7_label.csv')
Y = Y.values

Y = Y[:, 1:]
# Y = Y.transpose()
Y = np.ravel(Y)

Xnew1 = X[:, 0:48]
Xnew2 = X[:, 48:88]
Xnew3 = X[:, 88:342]
Xnew4 = X[:, 342:2326]
Xnew5 = X[:, 2326:2838]
Xnew6 = X[:, 2838:]

for r in range(3):
    if r == 0:
        R = 0.3
    elif r == 1:
        R = 0.5
    else:
        R = 0.7
    testfile = open("WeightedMVRFDISCal7%f.txt" % R, 'w')
    big = 0
    mm = ""
    err = 0
    train_in, test_in = splitdata(X=X, Y=Y, ratio=R, seed=1000)
    trainx = X[train_in]
    trainy = Y[train_in]
    testx=X[test_in]
    testy=Y[test_in]


    emv = []
    e61 = []
    e62 = []
    onndis_e = []
    tnndis_e = []
    ldadis_e = []
    lsvmdis_e = []
    nsvmdis_e = []
    for l in range(10):
        train_indices, test_indices = splitdata(X=X, Y=Y, ratio=R, seed=1000 + l)

        #view total
        X_features_train, X_features_test, wei= rf_dis(n_trees=500,  X=X,  train_indices=train_indices,test_indices=test_indices,seed=1000+l)

        m62 = RandomForestClassifier(n_estimators=500, random_state=1000 + l, oob_score=True, n_jobs=1).fit(
            X_features_train, Y[train_indices])
        e62.append(m62.score(X_features_test, Y[test_indices]))
        print(l)

        #Multi view RF dis

        X_features_train1, X_features_test1, w1 = rf_dis(n_trees=500, X=Xnew1, train_indices=train_indices,
                                                         test_indices=test_indices, seed=1000 + l)
        X_features_train2, X_features_test2, w2 = rf_dis(n_trees=500, X=Xnew2, train_indices=train_indices,
                                                         test_indices=test_indices, seed=1000 + l)
        X_features_train3, X_features_test3, w3 = rf_dis(n_trees=500, X=Xnew3, train_indices=train_indices,
                                                         test_indices=test_indices, seed=1000 + l)
        X_features_train4, X_features_test4, w4 = rf_dis(n_trees=500, X=Xnew4, train_indices=train_indices,
                                                         test_indices=test_indices, seed=1000 + l)
        X_features_train5, X_features_test5, w5 = rf_dis(n_trees=500, X=Xnew5, train_indices=train_indices,
                                                         test_indices=test_indices, seed=1000 + l)
        X_features_train6, X_features_test6, w6 = rf_dis(n_trees=500, X=Xnew6, train_indices=train_indices,
                                                         test_indices=test_indices, seed=1000 + l)
        X_features_trainm = (
                                w1 * X_features_train1 + w2 * X_features_train2 + w3 * X_features_train3 + w4 * X_features_train4 + w5 * X_features_train5 + w6 * X_features_train6) / (
                                w1 + w2 + w3 + w4 + w5 + w6)
        X_features_testm = (
                               w1 * X_features_test1 + w2 * X_features_test2 + w3 * X_features_test3 + w4 * X_features_test4 + w5 * X_features_test5 + w6 * X_features_test6) / (
                               w1 + w2 + w3 + w4 + w5 + w6)
        mv = RandomForestClassifier(n_estimators=500, random_state=1000+l, oob_score=True, n_jobs=-1).fit(X_features_trainm,Y[train_indices])
        emv.append(mv.score(X_features_testm ,Y[test_indices]))

        # 1-NN dis
        onn_dis_err, aauc = knn(n_neb=1, train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                                test_y=Y[test_indices])
        onndis_e.append(onn_dis_err)


        # 10 NN
        tnn_dis_err, aauc = knn(n_neb=10, train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                                test_y=Y[test_indices])
        tnndis_e.append(tnn_dis_err)


        # lsvm dis
        c = Lsvm_patatune(train_x=X_features_trainm, train_y=Y[train_indices])
        lsvm_dis_err, aauc = Lsvm(c=c, train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                                  test_y=Y[test_indices])
        lsvmdis_e.append(lsvm_dis_err)

        # svm dis
        nsvm_dis_err, aauc = Nsvm(train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                                  test_y=Y[test_indices])
        nsvmdis_e.append(nsvm_dis_err)

        print(l)
        print(R)

    testfile.write(str(e62) + '\n')
    testfile.write(str(emv) + '\n')
    testfile.write(str(onndis_e) + '\n')
    testfile.write(str(tnndis_e) + '\n')
    testfile.write(str(lsvmdis_e) + '\n')
    testfile.write(str(nsvmdis_e) + '\n')
    testfile.write("Single view RF dis:%f  \p %f " % ( np.mean(e62), np.std(e62)) + '\n')
    testfile.write("Multi view RF dis:%f  \p %f " % (np.mean(emv), np.std(emv)) + '\n')
    testfile.write("M view RF dis 1nn:%f  \p %f " % (np.mean(onndis_e), np.std(onndis_e)) + '\n')
    testfile.write("M view RF dis 10nn:%f  \p %f " % (np.mean(tnndis_e), np.std(tnndis_e)) + '\n')
    testfile.write("M view RF dis lsvm:%f  \p %f " % (np.mean(lsvmdis_e), np.std(lsvmdis_e)) + '\n')
    testfile.write("M view RF dis nsvm:%f  \p %f " % (np.mean(nsvmdis_e), np.std(nsvmdis_e)) + '\n')
    testfile.close()
