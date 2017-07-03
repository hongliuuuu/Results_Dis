from sklearn.externals.six import StringIO
import pydotplus as pydot
import numpy as np
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
import xlrd
from sklearn import tree
from sklearn.neighbors import KNeighborsClassifier
import random
import pandas
from sklearn.metrics import accuracy_score
from sklearn.model_selection import LeaveOneOut
from joblib import Parallel, delayed
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
def rf_dis(n_trees, X,Y,train_indices,test_indices,seed):
    clf = RandomForestClassifier(n_estimators=n_trees,
                                 random_state=seed, oob_score=True, n_jobs=-1)
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

def knn(n_neb, train_x, train_y, test_x, test_y):
    clf =KNeighborsClassifier(n_neighbors=n_neb, n_jobs=-1)
    clf.fit(train_x, train_y)
    test_error = clf.score(test_x, test_y)
    test_auc = clf.predict_proba(test_x)
    return test_error

def onn(test_x, train_y, test_y):
    n_s = test_x.shape[0]
    l = []
    for i in range(n_s):
        li = test_x[i]
        min = np.min(li)
        #find the positition of min
        p = li.tolist().index(min)
        l.append(train_y[p])
    s = accuracy_score(test_y, l)
    return s
def RF(n_trees,  seed, train_x, train_y, test_x, test_y):
    clf = RandomForestClassifier(n_estimators=n_trees,
                                  random_state = seed, oob_score=True)
    clf = clf.fit(train_x,train_y)
    oob_error = 1 - clf.oob_score_
    test_error = clf.score(test_x,test_y)
    test_auc = clf.predict_proba(test_x)
    #filename = './tmp1/RF_%d_.pkl'%seed
    #_ = joblib.dump(clf, filename, compress=9)
    return test_error
'''
X = [[0,0.2,0.6,0.8],
     [0.2,0,0.65,0.7],
     [0.6,0.65,0,0.1],
     [0.8,0.7,0.1,0]]
Y = [0,0,1,1]
test_x = [[0.2,0.1,0.5,0.6],
          [0.6,0.5,0.4,0.1]]
test_x = np.array(test_x)
Y = np.array(Y)
t_Y = Y.transpose()
test_y = [0,1]
test_y = np.array(test_y)
test_y = test_y.transpose()
print(onn(test_x,t_Y,test_y))
'''
url = 'text_pr_1.csv'
dataframe = pandas.read_csv(url, header=None)
array = dataframe.values
X = array
Y = pandas.read_csv('label_IDHCodel.csv', header=None)
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

Xnew1 = X[:, 0:1680]
Xnew2 = X[:, 1680:3360]
Xnew3 = X[:, 3360:5040]
Xnew4 = X[:, 5040:6720]
Xnew5 = X[:, 6720:6745]


data = []
err1 = []
err2 = []
err3 = []
err4 = []
tr = []

loo = LeaveOneOut()
def mcode(ite):
    if ite ==0:
        file = "30nIDHV1.csv"
        dataX = Xnew1
    if ite == 1:
        file = "30nIDHV2.csv"
        dataX = Xnew2
    if ite == 2:
        file = "30nIDHV3.csv"
        dataX = Xnew3
    if ite == 3:
        file = "30nIDHV4.csv"
        dataX = Xnew4
    if ite == 4:
        file = "30nIDHV5.csv"
        dataX = Xnew5
    for  i in range(50):
        ntrees = 20*(i+1)
        seed = 1000+i
        err = []
        errr = []
        eknn = []
        erf = []
        se = 1000
        for ite in range(30):
            se = se+1
            train_in, test_in = splitdata(dataX,Y,0.7,se)

            X_features_train1, X_features_test1, w1, pred1 = rf_dis(n_trees=ntrees, X=dataX, Y=Y, train_indices=train_in,
                                                                test_indices=test_in, seed=se)
            e = onn( train_y=Y[train_in],test_x=X_features_test1,test_y=Y[test_in])
            e2 = knn(n_neb=1, train_x=X_features_train1, train_y=Y[train_in], test_x=X_features_test1, test_y=Y[test_in])
            e3 = RF(n_trees=500,seed=se,train_x=X_features_train1, train_y=Y[train_in], test_x=X_features_test1, test_y=Y[test_in])
            #print(i,se,ntrees)
            e1 = w1
            err.append(e)
            errr.append(e1)
            eknn.append(e2)
            erf.append(e3)
        print(len(err),len(errr),len(eknn),len(erf))
        err1.append(np.mean(err))
        err2.append(np.mean(errr))
        err3.append(np.mean(eknn))
        err4.append(np.mean(erf))
        tr.append(ntrees)
    data.append(err1)
    data.append(err2)
    data.append(err3)
    data.append(err4)
    data.append(tr)
    prediction = pandas.DataFrame(data).to_csv(file)


if __name__ == '__main__':
    Parallel(n_jobs=5)(delayed(mcode)(ite=i) for i in range(5))
'''
pl.xlabel('tree number')

pl.plot(tr, err1)# use pylab to plot x and y
pl.plot(tr, err2)# use pylab to plot x and y
pl.plot(tr, err3)# use pylab to plot x and y
pl.plot(tr, err4)# use pylab to plot x and y
pl.legend(['1nn', 'RF', "Dis1NN"], loc='upper left')
pl.show()# show the plot on the screen'''


'''
clf = RandomForestClassifier(n_estimators=500,
                                 random_state=1000, oob_score=True, n_jobs=-1)
clf.fit(X, Y)
trees = clf.estimators_
ld = [estimator.tree_.max_depth for estimator in clf.estimators_]
print(np.max(ld))
dot_data = StringIO()
tree.export_graphviz(trees[3], out_file=dot_data,

filled=True, rounded=True,proportion =True,
special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
graph.write_pdf('BBC2.pdf')
'''