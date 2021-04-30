import pandas as pd
from scipy.io import arff
from sklearn import preprocessing
import numpy as np
import math
from random import shuffle

def loadData(filename):
    print( "Using dataset: "+filename)
    if '.arff' in filename:
        data = arff.loadarff(filename)
        dataset = pd.DataFrame(data[0]).values
        normalizeData(dataset)
        #dataset[:,range(len(dataset[0])-1)] = preprocessing.normalize(dataset[:,range(len(dataset[0])-1)])
    elif '.csv' in filename or '.data' in filename or '.all-data' in filename or '.data.Z' in filename:
        df=pd.read_csv(filename, sep=',',header=None)
        df = df.drop(columns=df.columns[(df == '?').any()])
        # df.loc[:, ~(df == 'Salty').any()]
        dataset = df.values
        normalizeData(dataset)
        #dataset[:,range(len(dataset[0])-1)] = preprocessing.normalize(dataset[:,range(len(dataset[0])-1)])
    else:
        print("Unknown format!!error")

    return dataset

def normalizeData(dataset):
    max = list(dataset[0])
    min = list(dataset[0])

    for x in range(len(dataset)):
        for y in range(len(dataset[x])-1):
            if isinstance(dataset[x][y],float) == True:
                if dataset[x][y] > max[y]:
                    max[y] = dataset[x][y]
                if dataset[x][y] < min[y]:
                    min[y] = dataset[x][y]

    for x in range(len(dataset)):
        for y in range(len(dataset[x])-1):
            if isinstance(dataset[x][y],float) == True:
                dataset[x][y] = (dataset[x][y] - min[y])/ (max[y]-min[y])