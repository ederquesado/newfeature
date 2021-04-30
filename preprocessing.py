import pandas as pd
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def pre_processing(dataset):
    # embaralhamento para distanciar amostras de dados semelhantes
    data_sf = shuffle(dataset, random_state=100)
    #dataset_sf = data_sf.reshape

    # atributos e classe
    X = data_sf[:,0:-1]
    Y = data_sf[:,-1]

    # normalizacao
    # normalizer = StandardScaler()
    # X_norm = normalizer.fit_transform(X)

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, shuffle=False)
    X_train = pd.DataFrame(X_train)
    X_test = pd.DataFrame(X_test)
    Y_train = Y_train
    Y_test = Y_test
    return X_train, X_test, Y_train, Y_test