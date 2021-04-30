import GetAcc


datasets_UCI = ['datasets/ionosphere.data',
                'datasets/lung-cancer.data',
                'datasets/sonar.all-data',
                'datasets/soybean-small.data',
                #'datasets/splice.data', #material genetico, não entendido como trabalhar com ele
                'datasets/waveform.data']


for i in range(len(datasets_UCI)):
    filename = datasets_UCI[i]
    GetAcc.testSFS(filename)
    GetAcc.testKfold(filename)





