import numpy as np
import warnings
import tensorflow as tf
import gpflow
from sklearn.metrics import accuracy_score

from helper_functions import helpers

np.random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings("ignore")


num_samples_training = [25, 50, 101, 150, 200, 400]
num_test = 10
print("Test Samples: "+str(num_test))
_, _, x_test, y_test = helpers.load_mnist_data(num_train_samples=1, num_test_samples=num_test)
for num_train in num_samples_training:
    print("Train Samples: "+str(num_train))
    x_train, y_train, _, _ = helpers.load_mnist_data(num_train_samples=num_train, num_test_samples=1)
    phi_x_train, phi_x_test = helpers.create_exponential_wassertein_feature_vector(x_train, x_test)

    print("PHIX Shape", phi_x_train.shape, phi_x_test.shape)

    phi_x_train, phi_x_test = phi_x_train.astype(np.float64), phi_x_test.astype(np.float64)
    y_train, y_test = y_train.astype(np.float64), y_test.astype(np.float64)

    # number of classes
    C = 10
    print("Num Classes = " + str(C))

    # k(x, y) = ( variance * (x.T@y) + offset )^ degree
    # For our case, we set offset=0 (non-trainable), resulting in approximation of theta*exp(-d_W^2/sigma^2)

    # define model
    model = gpflow.models.VGP(data=(phi_x_train, y_train),
                              mean_function=gpflow.mean_functions.Zero(),
                              kernel=gpflow.kernels.RBF(),
                              likelihood=gpflow.likelihoods.MultiClass(num_classes=C,
                                                                       invlink=gpflow.likelihoods.RobustMax(
                                                                           num_classes=C)),
                              num_latent_gps=C)

    # set trainable parameters
    # gpflow.utilities.set_trainable(model.kernel.degree, True)
    # gpflow.utilities.set_trainable(model.kernel.variance, True)
    # gpflow.utilities.set_trainable(model.kernel.offset, False)
    gpflow.utilities.set_trainable(model.likelihood.invlink.epsilon, False)
    gpflow.utilities.print_summary(model)

    # running optimization
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        model.training_loss_closure(),
        model.trainable_variables
    )

    # see model parameters after training
    gpflow.utilities.print_summary(model)

    # Make Predictions
    y_test_mean_predicted, y_test_var_predicted = model.predict_f(phi_x_test)
    y_predicted = tf.argmax(y_test_mean_predicted, axis=1)

    print(y_predicted)
    print(y_test)

    print("Classification Accuracy: " + str(accuracy_score(y_true=y_test, y_pred=y_predicted)))





