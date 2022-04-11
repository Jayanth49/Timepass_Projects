import numpy as np
import gpflow.ci_utils
import tensorflow as tf
from data_loader import loader

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)
tf.random.set_seed(42)


def pre_process_data(xtrain, ytrain, xtest, ytest, standardScale=False, jitter=False, asFloat64=False):
    if jitter:
        xtest, xtrain = xtest + 1e-6, xtrain + 1e-6
    if standardScale:
        scaler = StandardScaler()
        scaler.fit(xtrain)
        xtrain, xtest = scaler.transform(xtrain), scaler.transform(xtest)
    if asFloat64:
        xtrain, ytrain, xtest, ytest = list(map(lambda x: x.astype(np.float64), [xtrain, ytrain, xtest, ytest]))

    return xtrain, ytrain, xtest, ytest


num_training_per_class = [2, 5, 10, 20, 30, 40, 50]
num_test_per_class = 50
print("Num Test Samples Per Class: " + str(num_test_per_class))
_, _, x_test, y_test = loader.data_loader(dataset="mnist", dataset_type="partial",
                                          num_each_class_train=1, num_each_class_test=num_test_per_class,
                                          seed=42)
for num_train in num_training_per_class:
    print("Train Samples Per Class: " + str(num_train))
    x_train, y_train, _, _ = loader.data_loader(dataset="mnist", dataset_type="partial",
                                                num_each_class_train=num_train, num_each_class_test=1,
                                                seed=42)

    # pre process the data
    x_train, y_train, x_test, y_test = pre_process_data(x_train, y_train, x_test, y_test,
                                                        standardScale=False, jitter=True, asFloat64=True)

    # number of classes
    C = 10
    print("Num Classes = " + str(C))

    # initialize lengthscales of RBF Kernel
    length_scales = [np.std(x_train[:, i], ddof=1)+1e-4 for i in range(x_train.shape[1])]
    # print("length_scales: "+ str(len(length_scales)))
    # length_scales = [0.1] * x_train.shape[1]

    # define model
    model = gpflow.models.VGP(data=(x_train, y_train),
                              mean_function=gpflow.mean_functions.Zero(),
                              kernel=gpflow.kernels.SquaredExponential(lengthscales=length_scales),
                              likelihood=gpflow.likelihoods.MultiClass(num_classes=C,
                                                                       invlink=gpflow.likelihoods.RobustMax(
                                                                           num_classes=C)),
                              num_latent_gps=C)

    # set trainable parameters
    gpflow.utilities.set_trainable(model.kernel.variance, True)
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
    # print("Optimization Log:\n", opt_logs)
    # gpflow.utilities.print_summary(model)

    # Make Predictions
    x_test_mean_predicted, x_test_var_predicted = model.predict_f(x_test)
    # print(x_test_mean_predicted)
    # break
    y_predicted = tf.argmax(x_test_mean_predicted, axis=1)

    print("y_predicted: ",'\n',y_predicted)
    print("y_test: ",'\n',y_test)
    # print(y_test)

    print("Classification Accuracy: " + str(accuracy_score(y_true=y_test, y_pred=y_predicted)))
