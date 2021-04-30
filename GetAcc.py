import numpy as np
from sklearn.svm import SVC
import preprocessing
import misc
from feature_selection import sfs_lw
import matplotlib.pyplot as plt
import time
from sklearn.model_selection import StratifiedKFold, cross_validate, cross_val_score, cross_val_predict
from sklearn.metrics import make_scorer, accuracy_score



def testSFS(filename):
    print("--------------------LW---------------------")
    accuracy = []
    tempo = []
    dataset = misc.loadData(filename)
    dataset = np.asarray(dataset)
    X_train, X_test, Y_train, Y_test = preprocessing.pre_processing(dataset)
    feature_selection = sfs_lw(X_train, Y_train)
    clf_svm = SVC(C=128, kernel='linear', decision_function_shape='ovo')

    for i in range(len(feature_selection)):
        start_time = time.time()
        subset = feature_selection[0:i + 1]
        features = X_train[subset]
        clf_svm.fit(features, Y_train)
        clf_svm.predict(features)
        accuracy.append(clf_svm.score(X_test[subset], Y_test))
        tempo.append(time.time() - start_time)

    print(f"Acuracia média {(sum(accuracy)/len(accuracy))*100}")
    print(f"Tempo média {(sum(tempo) / len(tempo)) *1000}")
    # plt.figure(figsize=(14, 8))
    # plt.plot(range(len(feature_selection)), accuracy, color='blue', linestyle='dashed', marker='o')
    # plt.xlabel('feature_selection')
    # plt.ylabel('accuracy')
    # plt.show()


def testKfold(filename):
    print("-------------------CV-------------------------")
    accuracy = []
    tempo = []
    n_splits = 5

    dataset = misc.loadData(filename)
    dataset = np.asarray(dataset)
    clf_svm = SVC(C=128, kernel='linear', decision_function_shape='ovo')

    for i in range(5):
        X = dataset[:, 0:-1]
        Y = dataset[:, -1]
        start_time = time.time()

        X_train, X_test, Y_train, Y_test = preprocessing.pre_processing(dataset)
        kf = StratifiedKFold(n_splits=n_splits, shuffle=True)
        clf_svm.fit(X_train, Y_train)

        cross_val = cross_validate(clf_svm, X, Y, cv=kf)
        cross_val_predicted = cross_val_predict(clf_svm, X, Y, cv=kf)
        cross_score_accuracy = cross_val_score(clf_svm, X, Y, cv=kf, scoring=make_scorer(accuracy_score))
        accuracy.append(cross_score_accuracy.mean() * 100)

        tempo.append(time.time() - start_time)

    print(f"Acuracia média {sum(accuracy)/len(accuracy)}")
    print(f"Tempo média {(sum(tempo) / len(tempo)) *1000}")
    # plt.figure(figsize=(14, 8))
    # plt.plot(range(len(feature_selection)), accuracy, color='blue', linestyle='dashed', marker='o')
    # plt.xlabel('feature_selection')
    # plt.ylabel('accuracy')
    # plt.show()