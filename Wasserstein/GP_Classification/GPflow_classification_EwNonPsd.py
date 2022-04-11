import gpflow
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score

from helper_functions import helpers, gpflow_new_kernels

np.random.seed(42)
tf.random.set_seed(42)


def pre_process_data(xtrain, ytrain, xtest, ytest, mapToProbability=True, standardScale=True, jitter=True, asFloat64=True):
    if jitter:
        xtest, xtrain = xtest + np.random.uniform(low=1e-7, high=1e-5, size=xtest.shape), xtrain + np.random.uniform(low=1e-7, high=1e-5, size=xtrain.shape)
    if mapToProbability:
        xtrain, xtest = helpers.map_to_probability_simplex([xtrain, xtest])
    if standardScale:
        scaler = StandardScaler()
        scaler.fit(xtrain)
        xtrain, xtest = scaler.transform(xtrain), scaler.transform(xtest)
    if asFloat64:
        xtrain, ytrain, xtest, ytest = list(map(lambda x: x.astype(np.float64), [xtrain, ytrain, xtest, ytest]))

    return xtrain, ytrain, xtest, ytest


# MNIST DATA
nsx, nsy = 28, 28
m = helpers.get_image_cost_matrix(nsx, nsy)
num_samples_training = [25, 50, 101, 150, 200, 400]
num_test = 25

print("Num Test Samples: " + str(num_test))
_, _, x_test, y_test = helpers.load_mnist_data(num_train_samples=1, num_test_samples=num_test)

for num_train in num_samples_training:
    print("Train Samples: " + str(num_train))
    x_train, y_train, _, _ = helpers.load_mnist_data(num_train_samples=num_train, num_test_samples=1)

    # preprocess the data
    x_train, y_train, x_test, y_test = pre_process_data(x_train, y_train, x_test, y_test,
                                                        standardScale=False, mapToProbability=True,
                                                        jitter=True, asFloat64=True)
    # print(type(x_test))
    # x_test += 1e-6

    m = helpers.get_image_cost_matrix(28, 28)
    C = 10
    model = gpflow.models.VGP(data=(x_train, y_train),
                              mean_function=gpflow.mean_functions.Zero(),
                              kernel=gpflow_new_kernels.EWNonPsd(cost_matrix=m,
                                                                 initial_variance=1.0, initial_scale=1.0,
                                                                 reg=0.4, stopThr=5e-2, numIterMax=1000,
                                                                 xtrain=x_train, xtest=x_test),
                              likelihood=gpflow.likelihoods.MultiClass(num_classes=C,
                                                                       invlink=gpflow.likelihoods.RobustMax(
                                                                           num_classes=C)),
                              num_latent_gps=C)

    # set trainable parameters
    gpflow.utilities.set_trainable(model.kernel.variance, True)
    # gpflow.utilities.set_trainable(model.kernel.scale, False)
    gpflow.utilities.set_trainable(model.likelihood.invlink.epsilon, False)

    # see model parameters after training
    gpflow.utilities.print_summary(model)

    # running optimization
    opt = gpflow.optimizers.Scipy()
    opt_logs = opt.minimize(
        model.training_loss_closure(compile=False),
        model.trainable_variables,
        compile=False
    )

    # see model parameters after training
    gpflow.utilities.print_summary(model)

    # Make Predictions
    x_test_mean_predicted, x_test_var_predicted = model.predict_f(x_test)
    y_predicted = tf.argmax(x_test_mean_predicted, axis=1)
    print(y_predicted)
    print(y_test)
    print("Classification Accuracy: " + str(accuracy_score(y_true=y_test, y_pred=y_predicted)))
