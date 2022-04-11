import numpy as np
from helper_functions import helpers
from sklearn.svm import SVC
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


num_samples_training = [10, 20, 25]
num_test = 10
_, _, x_test, y_test = helpers.load_mnist_data(num_train_samples=1, num_test_samples=num_test)
for num_train in num_samples_training:
    print("Train Samples: "+str(num_train))
    x_train, y_train, _, _ = helpers.load_mnist_data(num_train_samples=num_train, num_test_samples=1)
    phi_x_train, phi_x_test = helpers.create_exponential_wassertein_feature_vector(x_train=x_train, x_test=x_test, normalize=True)
    clf1 = SVC(C=1.0, kernel='linear', shrinking=False, probability=False,
              tol=1e-7, cache_size=1000, class_weight=None, verbose=False,
              max_iter=-1, decision_function_shape='ovo', random_state=42)
    clf2 = SVC(C=1.0, kernel='rbf', shrinking=False, probability=False,
              tol=1e-7, cache_size=1000, class_weight=None, verbose=False,
              max_iter=-1, decision_function_shape='ovo', random_state=42)
    exp_wassertein_score = helpers.get_classification_score(clf1, phi_x_train, y_train, phi_x_test, y_test)
    rbf_score = helpers.get_classification_score(clf2, x_train, y_train, x_test, y_test)
    print(" ExpWasserteinAccuracy="+str(exp_wassertein_score)+" RBFAccuracy="+str(rbf_score))
