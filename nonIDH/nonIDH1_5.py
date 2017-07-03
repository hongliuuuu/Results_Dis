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

def rf_dis(n_trees, X,Y,train_indices,test_indices,seed):
    clf = RandomForestClassifier(n_estimators=500,
                                 random_state=seed, oob_score=True, n_jobs=1)
    clf = clf.fit(X[train_indices], Y[train_indices])
    pred = clf.predict(X[test_indices])
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
    return X_features3[train_indices],X_features3[test_indices],weight,pred

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
    print(clf.score(test_x,test_y))
    return clf.best_params_['C']
def Lsvm_patatune(train_x,train_y):
    tuned_parameters = [
        {'kernel': ['linear'], 'C': [0.01,0.1, 1, 10, 100, 1000]}]
    clf = GridSearchCV(SVC(C=1, probability=True), tuned_parameters, cv=5, n_jobs=1
                       )  # SVC(probability=True)#SVC(kernel="linear", probability=True)
    clf.fit(train_x, train_y)
    return clf.best_params_['C']
"""
url = 'nus_1.csv'
dataframe = pandas.read_csv(url)  # , header=None)
array = dataframe.values
X = array[:, 1:]

for i in range(4):
    url = 'nus_' + str(i + 2) + '.csv'
    dataframe = pandas.read_csv(url)  # , header=None)
    array = dataframe.values
    X1 = array[:, 1:]
    X = np.concatenate((X, X1), axis=1)
Y = pandas.read_csv('nus_label.csv')
Y = Y.values

Y = Y[:, 1:]
# Y = Y.transpose()
Y = np.ravel(Y)
train_indices, test_indices = splitdata(X=X, Y=Y, ratio=0.1, seed=1000 )

n_trees = 500
n_feat = selected_f(639)  # features selecleted
n_features = X.shape[1]
for i in range(n_features):
    s = X[:,i]
    mn = np.max(s)-np.min(s)
    if mn == 0:
        print(i)
"""


def mcode(ite):
    R = 0.5
    testfile = open("nonIDH1%f_%f.txt" % (R,ite), 'w')

    e11 = []
    e12 = []
    e21 = []
    e22 = []
    e31 = []
    e32 = []
    e41 = []
    e42 = []
    e51 = []
    e52 = []
    e62 = []
    e66 = []
    e67 = []
    e72 = []
    e8 = []
    erelf = []
    elfs = []
    erfsvm = []
    elaterf = []
    elaterfdis = []
    esv = []
    esvd = []

    seed = 1000 + ite
    #data reading
    url = 'text_nonIDH1_1.csv'
    dataframe = pandas.read_csv(url, header=None)
    array = dataframe.values
    X = array
    Y = pandas.read_csv('label_nonIDH1.csv', header=None)
    Y = Y.values
    Y = np.ravel(Y)
    print(Y.shape)

    for i in range(5):
        url = 'text_nonIDH1_' + str(i + 2) + '.csv'
        dataframe = pandas.read_csv(url, header=None)
        array = dataframe.values
        X1 = array
        print(X1.shape)
        X = np.concatenate((X, X1), axis=1)

    Xnew1 = X[:, 0:1680]
    Xnew2 = X[:, 1680:3360]
    Xnew3 = X[:, 3360:5040]
    Xnew4 = X[:, 5040:6720]
    Xnew5 = X[:, 6720:6745]
    Xnew6 = X[:, 6745:]
    #X = np.concatenate((Xnew1, Xnew2, Xnew3, Xnew4, Xnew5), axis=1)
    n_features = X.shape[1]
    n_samples = X.shape[0]
    print("datasize")
    print(X.shape)
    n_trees = 500
    n_feat = selected_f(n_features)  # features selecleted
    train_indices, test_indices = splitdata(X=X, Y=Y, ratio=R, seed=seed)
    train_x = X[train_indices]
    train_y = Y[train_indices]
    test_x = X[test_indices]
    test_y = Y[test_indices]

    # RELF
    print("Start RELF")
    relfind = relf(n_neb=int(n_samples), n_feat=n_feat, trainx=train_x, trainy=train_y, testx=test_x)
    train_x = X[train_indices]
    train_y = Y[train_indices]
    TX = X[train_indices]
    TEX = X[test_indices]
    relftrainX = TX[:, relfind]
    relftestX = TEX[:, relfind]

    relf_err, aauc = RF(n_trees=n_trees, seed=seed, train_x=relftrainX, train_y=train_y, test_x=relftestX,
                        test_y=test_y)
    erelf.append(relf_err)
    prediction = pandas.DataFrame(aauc).to_csv('./pre/NUSRELFR%fpre%d.csv' % (R, ite))
    #SVMRFE
    print("Start SVMRFE")
    min_max_scaler = preprocessing.MinMaxScaler()
    svm_X = min_max_scaler.fit_transform(X)
    svmtrain_x = svm_X[train_indices]
    svmtest_x = svm_X[test_indices]
    c = Lsvm_patatune(train_x=svmtrain_x, train_y=train_y)
    # lfs error
    lfs_train_X, lfs_test_X = lsvm_rfe(c=c, n_feat=n_feat, trainX=svmtrain_x, trainy=train_y, testX=svmtest_x)

    lfs_err, aauc = RF(n_trees=n_trees, seed=seed, train_x=lfs_train_X, train_y=train_y, test_x=lfs_test_X,
                       test_y=test_y)
    elfs.append(lfs_err)
    prediction = pandas.DataFrame(aauc).to_csv('./pre/NUSSVMRFER%fpre%d.csv' % (R, ite))
    print("Start rest")
    #view1

    X_features_train1, X_features_test1,w1,pred1= rf_dis(n_trees=500,  X=Xnew1,Y=Y,  train_indices=train_indices,test_indices=test_indices,seed=seed)
    m12 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_train1, Y[train_indices])
    pre1 = m12.predict(X_features_test1)
    e12.append(m12.score(X_features_test1, Y[test_indices]))
    e11.append(w1)
    #view 2

    X_features_train2, X_features_test2, w2,pred2 = rf_dis(n_trees=500, X=Xnew2,Y=Y, train_indices=train_indices,
                                                           test_indices=test_indices, seed=seed)
    m22 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_train2, Y[train_indices])
    pre2 = m22.predict(X_features_test2)
    e22.append(m22.score(X_features_test2, Y[test_indices]))
    e21.append(w2)

    #view 3

    X_features_train3, X_features_test3, w3,pred3 = rf_dis(n_trees=500, X=Xnew3,Y=Y, train_indices=train_indices,
                                                           test_indices=test_indices, seed=seed)
    m32 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_train3, Y[train_indices])
    pre3 = m32.predict(X_features_test3)
    e32.append(m32.score(X_features_test3, Y[test_indices]))
    e31.append(w3)

    #view 4

    X_features_train4, X_features_test4, w4,pred4 = rf_dis(n_trees=500, X=Xnew4,Y=Y, train_indices=train_indices,
                                                           test_indices=test_indices, seed=seed)
    m42 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_train4, Y[train_indices])
    pre4 = m42.predict(X_features_test4)
    e42.append(m42.score(X_features_test4, Y[test_indices]))
    e41.append(w4)

    # view 5

    X_features_train5, X_features_test5, w5, pred5 = rf_dis(n_trees=500, X=Xnew5, Y=Y, train_indices=train_indices,
                                                            test_indices=test_indices, seed=seed)
    m52 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_train5, Y[train_indices])
    pre5 = m52.predict(X_features_test5)
    e52.append(m52.score(X_features_test5, Y[test_indices]))
    e51.append(w5)

    # view 6

    X_features_train6, X_features_test6, w6, pred6 = rf_dis(n_trees=500, X=Xnew6, Y=Y, train_indices=train_indices,
                                                            test_indices=test_indices, seed=seed)
    m62 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_train6, Y[train_indices])
    pre6 = m62.predict(X_features_test6)
    e66.append(m62.score(X_features_test6, Y[test_indices]))
    e67.append(w6)

    # Late RF
    resall1 = np.column_stack((pred1, pred2, pred3, pred4,pred5,pred6))
    Laterf = list(range(len(test_indices)))
    for i in range(len(test_indices)):
        Laterf[i], empty = Counter(resall1[i]).most_common()[0]
    LRF = accuracy_score(Y[test_indices], Laterf)
    elaterf.append(LRF)
    # Late RF dis
    resall = np.column_stack((pre1, pre2, pre3, pre4,pre5,pre6))
    LSVTres = list(range(len(test_indices)))
    for i in range(len(test_indices)):
        LSVTres[i], empty = Counter(resall[i]).most_common()[0]
    LSVTscore = accuracy_score(Y[test_indices], LSVTres)
    elaterfdis.append(LSVTscore)
    #single view total
    X_features_train, X_features_test,w6 , pred6= rf_dis(n_trees=500, X=X,Y=Y, train_indices=train_indices,
                                                         test_indices=test_indices, seed=seed)

    m62 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_train, Y[train_indices])
    esvd.append(m62.score(X_features_test, Y[test_indices]))
    esv.append(w6)

    # Concatenanted multiview
    Xconcate_train = np.concatenate((X_features_train1,X_features_train2,X_features_train3,X_features_train4,X_features_train5,X_features_train6),axis=1)
    Xconcate_test = np.concatenate((X_features_test1,X_features_test2,X_features_test3,X_features_test4,X_features_test5,X_features_test6),axis=1)
    m72 = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        Xconcate_train, Y[train_indices])
    e72.append(m72.score(Xconcate_test, Y[test_indices]))



    #multi view
    X_features_trainm = (
                            X_features_train1 + X_features_train2 + X_features_train3 + X_features_train4+X_features_train5 +X_features_train6) / 6
    X_features_testm = (
                           X_features_test1 + X_features_test2 + X_features_test3 + X_features_test4 +X_features_test5+X_features_test6) / 6
    mv = RandomForestClassifier(n_estimators=500, random_state=seed, oob_score=True, n_jobs=1).fit(
        X_features_trainm, Y[train_indices])
    e8.append(mv.score(X_features_testm, Y[test_indices]))

    #RFSVM
    c = nLsvm_patatune(train_x=X_features_trainm, train_y=Y[train_indices], test_x=X_features_testm,
                       test_y=Y[test_indices])
    print(w1, w2, w3, w4,w5,w6)
    print(c)
    clf = SVC(C=c, kernel='precomputed')
    clf.fit(X_features_trainm, Y[train_indices])
    erfsvm.append(clf.score(X_features_testm, Y[test_indices]))

    testfile.write(str(e12) + '\n')
    testfile.write(str(e22) + '\n')
    testfile.write(str(e32) + '\n')
    testfile.write(str(e42) + '\n')
    testfile.write(str(e52) + '\n')
    testfile.write(str(e62) + '\n')


    #print(l,R)


    """
    testfile.write(str(e12) + '\n')
    testfile.write(str(e22) + '\n')
    testfile.write(str(e32) + '\n')
    testfile.write(str(e42) + '\n')
    testfile.write(str(e52) + '\n')
    testfile.write(str(e62) + '\n')
    """
    testfile.write("View 1 RF:%f pm%f " % (np.mean(e11), np.std(e11)) + '\n')
    testfile.write("View1 RF dis :%f\pm%f " % (np.mean(e12), np.std(e12)) + '\n')
    testfile.write("View 2 RF:%f pm %f " % (np.mean(e21), np.std(e21)) + '\n')
    testfile.write(":%f\pm%f " % ( np.mean(e22), np.std(e22)) + '\n')
    testfile.write("View 3 RF:%f\pm%f " % (np.mean(e31), np.std(e31)) + '\n')
    testfile.write(":%f\pm%f " % (np.mean(e32), np.std(e32)) + '\n')
    testfile.write("View 4 RF:%f\pm%f " % (np.mean(e41), np.std(e41)) + '\n')
    testfile.write(":%f\pm%f " % ( np.mean(e42), np.std(e42)) + '\n')
    testfile.write("View 5 RF:%f\pm%f " % (np.mean(e51), np.std(e51)) + '\n')
    testfile.write(":%f\pm%f " % (np.mean(e52), np.std(e52)) + '\n')
    testfile.write("View 6 RF:%f\pm%f " % (np.mean(e66), np.std(e66)) + '\n')
    testfile.write(":%f\pm%f " % (np.mean(e67), np.std(e67)) + '\n')

    testfile.write("AWA 0.1 RF:&%s pm%s &" % (floored_percentage(np.mean(esv),2), floored_percentage(np.std(esv),2)) + '\n')
    testfile.write(" RELF&%s pm%s & " % (floored_percentage(np.mean(erelf),2), floored_percentage(np.std(erelf),2)) + '\n')
    testfile.write(" SVMRFE&%s pm%s & " % (floored_percentage(np.mean(elfs),2), floored_percentage(np.std(elfs),2)) + '\n')
    testfile.write("RFSVM&%s pm%s & " % (floored_percentage(np.mean(erfsvm),2), floored_percentage(np.std(erfsvm),2)) + '\n')
    testfile.write("RFDIS &%s pm%s & " % (floored_percentage(np.mean(e8),2), floored_percentage(np.std(e8),2)) + '\n')
    testfile.write(" LATERF&%s pm%s &" % (floored_percentage(np.mean(elaterf),2), floored_percentage(np.std(elaterf),2)) + '\n')
    testfile.write(" LATERFDIS&%s pm%s & " % (floored_percentage(np.mean(elaterfdis),2), floored_percentage(np.std(elaterfdis),2)) + '\n')

    testfile.write("Single view rf dis:%f\pm%f " % ( np.mean(esvd), np.std(esvd)) + '\n')
    testfile.write("Concatenated multi view dis:%f\pm%f " % (np.mean(e72), np.std(e72)) + '\n')






    testfile.close()

if __name__ == '__main__':
    Parallel(n_jobs=4)(delayed(mcode)(ite=i) for i in range(10))