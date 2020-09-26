import machine
import numpy as np
from joblib import dump, load
from sklearn import svm
from sklearn.model_selection import train_test_split

DATA_FOLDER = '../data/'

class SVM(machine.machine):
    
    def __init__(self, dataName):
        self.dataPath = DATA_FOLDER + dataName
    
    def rbf_training(self):
        print("Runing SVM RBF training...")
        x_train, y_train = super().getDatasetTraining(self.dataPath)
        for c in [0.1, 1, 10]:
            for gamma in [0.1, 1, 10]:
                s = svm.SVC(C=c, kernel='rbf', gamma=gamma)
                s.fit(x_train, y_train)
                dump(s, 'svm_rbf_'+c+'_'+gamma+'.joblib')
                # pickle.dumps(s)
        # print('Saving model')

                # predicted = rbf_svm.predict(data_test[0])
                # print("SETTINGS:")
                # print("- C param: ", c)
                # print("- gamma param: ", gamma)
                # gen_metrics(gen_confusion_matrix(n_class, predicted, np.array(data_test[1])))
        