from svm import SVM
import sys
import utils


if __name__ == "__main__":
    dataName = utils.getArgs(sys.argv, '-i')
    svm = SVM(dataName)
    c_list = [1, 1, 0.05]
    g_list = [1, 0.05, 1]
    for c,g in zip(c_list,g_list):
        svm.rbf_training(c=c, gamma=g)
        result = svm.evalMetrics()