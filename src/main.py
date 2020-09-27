from svm import SVM
import sys
import utils


if __name__ == "__main__":
    dataName = utils.getArgs(sys.argv, '-i')
    svm = SVM(dataName)
    svm.rbf_training()