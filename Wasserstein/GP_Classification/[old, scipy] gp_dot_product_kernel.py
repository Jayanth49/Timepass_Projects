import numpy as np
import sklearn.gaussian_process
import warnings
from sklearn.gaussian_process import GaussianProcessClassifier
from helper_functions import helpers

np.random.seed(42)
warnings.filterwarnings("ignore")


num_samples_training = [100, 20, 25]
num_test = 10
print("Test Samples: "+str(num_test))
_, _, x_test, y_test = helpers.load_mnist_data(num_train_samples=1, num_test_samples=num_test)
for num_train in num_samples_training:
    print("Train Samples: "+str(num_train))
    x_train, y_train, _, _ = helpers.load_mnist_data(num_train_samples=num_train, num_test_samples=1)
    phi_x_train, phi_x_test = helpers.create_exponential_wassertein_feature_vector(x_train, x_test)

    gpc_1 = GaussianProcessClassifier(kernel=sklearn.gaussian_process.kernels.DotProduct(), random_state=42)
    accuracy_score_exp_wassertein_gp = helpers.get_classification_score(clf=gpc_1,
                                                                        x_train=phi_x_train, y_train=y_train,
                                                                        x_test=phi_x_test, y_test=y_test)
    print("Accuracy for Exp Wassertein GP with Dot Product kernel on Feature Space: ", accuracy_score_exp_wassertein_gp)

    gpc_2 = GaussianProcessClassifier(kernel=sklearn.gaussian_process.kernels.RBF(), random_state=42)
    accuracy_score_rbf_gp = helpers.get_classification_score(clf=gpc_2,
                                                             x_train=x_train, y_train=y_train,
                                                             x_test=x_test, y_test=y_test)
    print("Accuracy for GP with RBF Kernel on Original Space: ", accuracy_score_rbf_gp)






