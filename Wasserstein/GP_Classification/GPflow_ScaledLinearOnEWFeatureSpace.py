import numpy as np
import warnings
import tensorflow as tf
import gpflow
from sklearn.metrics import accuracy_score
from data_loader import loader
from helper_functions import helpers

np.random.seed(42)
tf.random.set_seed(42)
warnings.filterwarnings("ignore")

num_training_per_class = [2, 5, 10, 20, 30, 40, 50]
all_accuracies = []
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
    phi_x_train, phi_x_test = helpers.create_exponential_wassertein_feature_vector(x_train, x_test)

    print("PHIX Shape", phi_x_train.shape, phi_x_test.shape)

    phi_x_train, phi_x_test = phi_x_train.astype(np.float64), phi_x_test.astype(np.float64)
    y_train, y_test = y_train.astype(np.float64), y_test.astype(np.float64)

    # number of classes
    C = 10
    print("Num Classes = " + str(C))

    # define model
    model = gpflow.models.VGP(data=(phi_x_train, y_train),
                              mean_function=gpflow.mean_functions.Zero(),
                              kernel=gpflow.kernels.Linear(),
                              likelihood=gpflow.likelihoods.MultiClass(num_classes=C,
                                                                       invlink=gpflow.likelihoods.RobustMax(
                                                                           num_classes=C)),
                              num_latent_gps=C)

    # set trainable parameters
    # gpflow.utilities.set_trainable(model.kernel.theta, True)
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

    print(y_predicted[:10])
    print(y_test[:10])

    print("Classification Accuracy: " + str(accuracy_score(y_true=y_test, y_pred=y_predicted)))
    all_accuracies.append(accuracy_score(y_true=y_test, y_pred=y_predicted))
    # break

print("Num test per class = ", num_test_per_class)
print("Num Train Per Class = ", num_training_per_class)
print("All accuracies = ", all_accuracies)




