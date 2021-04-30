import numpy as np
import pandas as pd
import math

u = 1.0

# retorna o centroide de todas as classes
def centroid(X_fs, Y_train):
    Vc = []
    clusters = np.concatenate([X_fs, Y_train[:, None]], axis=1)
    for ci in np.unique(Y_train): # para cada classe
        cluster_i = clusters[clusters[:, -1] == ci]
        Xci = cluster_i[:, :-1]
        C_norm = len(Xci)
        Vc.append(Xci.sum() / C_norm)
    return pd.Series(Vc)

def euclidean_distance(inst1, inst2):
    #sub = inst1 - inst2
    sqr_sub = (inst1 - inst2)**2
    #sqr_sub = sub.apply(lambda x: x**2)
    dist = sqr_sub.sum()
    return math.sqrt(dist)


# retorna o raio de todas as classes
def class_radius(X_fs, Vc, Y_train):
    i = 0
    Rc = []
    clusters = np.concatenate([X_fs, Y_train[:, None]], axis=1)
    for ci in np.unique(Y_train):  # para cada classe
        cluster_i = clusters[clusters[:, -1] == ci]
        Xci = cluster_i[:, :-1]
        sum_dist = 0
        Vci = Vc[i]

        for k in range(len(Xci)):
            x_k = Xci[k]
            sum_dist += euclidean_distance(x_k, Vci)
        K = u * len(Xci)
        Rc.append(sum_dist / K)
        i += 1

    return pd.Series(Rc)


def lw_index(X_fs, Y_train):
    minFD_ij = []
    Vc = centroid(X_fs, Y_train)
    Rc = class_radius(X_fs, Vc, Y_train)
    C_fs = pd.concat([Vc, Rc], axis=1)
    #     FDs = []
    for i in range(len(C_fs)):
        FDs = []
        vc_i = C_fs.iloc[i][0]
        rc_i = C_fs.iloc[i][1]

        for j in range(len(C_fs)):
            vc_j = C_fs.iloc[j][0]
            rc_j = C_fs.iloc[j][1]
            FD_ij = euclidean_distance(vc_i, vc_j) - (rc_i + rc_j)
            FDs.append(FD_ij)
        minFD_ij.append(min(FDs))

    M = len(C_fs)  # quantidade de classes
    LWx = sum(minFD_ij) / M
    return LWx


def sfs_lw(Fo, Y_train):
    Fs = pd.DataFrame([])
    while(not Fo.empty):
        argLW = []
        for c in Fo:
            fc = pd.DataFrame(Fo[c])
            Fc = pd.concat([fc,Fs], axis=1)
            LWfc = lw_index(Fc, Y_train)
            argLW.append([LWfc,c])
        d = max(argLW)[1]
        fd = pd.DataFrame(Fo[d])
        Fd = pd.DataFrame(fd)
        Fs = pd.concat([Fs,Fd], axis=1)
        Fo = Fo.drop(Fd,axis=1)
    return Fs.columns
