import gpflow
import tensorflow as tf
import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from data_loader import loader
from helper_functions import helpers, gpflow_new_kernels
import matplotlib.pyplot as plt
import gpflow_new_kernels_modifed_2_Inverse_variance
import time

np.random.seed(4)
tf.random.set_seed(4)



def pre_process_data(xtrain, ytrain, xtest, ytest, mapToProbability=False, standardScale=False, jitter=False, asFloat64=False):
    np.random.seed(42)
    tf.random.set_seed(42)
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
num_training_per_class = [2]   
all_accuracies = []
num_test_per_class = 1
print("Num Test Samples Per Class: " + str(num_test_per_class)) 
_, _, x_test, y_test = loader.data_loader(dataset="mnist", dataset_type="partial",
                                          num_each_class_train=1, num_each_class_test=num_test_per_class,
                                          seed=42)
_, _, x_test, y_test = pre_process_data(xtrain=x_test.copy(), ytrain=y_test.copy(), xtest=x_test, ytest=y_test,
                                        standardScale=False, mapToProbability=True,
                                        jitter=True, asFloat64=True)

# # As test data is same for all of next iterations, precomputing wassertein distances
# precomputed_test_test_wassertein_distances = helpers.get_wassertein_distances(
#     cost_matrix=m, x1=x_test, label='Pre-computing Test Test Wassertein Distances')

precomputed_test_test_wassertein_distances = helpers.get_wassertein_distances(
    cost_matrix=m, x1=x_test, label='Pre-computing Test Test Wassertein Distances')

start = time.time()
for num_train in num_training_per_class:
    print("Train Samples Per Class: " + str(num_train))
    x_train, y_train, _, _ = loader.data_loader(dataset="mnist", dataset_type="partial",
                                                num_each_class_train=num_train, num_each_class_test=1,
                                                seed=42)
    # preprocess the data
    x_train, y_train, _, _ = pre_process_data(x_train, y_train, x_test, y_test,
                                                        standardScale=False, mapToProbability=True,
                                                        jitter=True, asFloat64=True)
    
    precomputed_train_train_wassertein_distances = helpers.get_wassertein_distances(
                    cost_matrix=m, x1=x_train, label='Pre-computing Train Train Wassertein Distances')
    precomputed_train_test_wassertein_distances = helpers.get_wassertein_distances(
                    cost_matrix=m, x1=x_test,x2 = x_train, label='Pre-computing Train Test Wassertein Distances')
    
    
    C = 10
    # model = gpflow.models.VGP(data=(x_train, y_train),
    #                           mean_function=gpflow.mean_functions.Zero(),
    #                           kernel=gpflow_new_kernels_modifed_2_Inverse_variance.ExpWassWithPsdApproximation(
    #                               cost_matrix=m,inverse_variance= None, initial_scale=1.0, reg=0.4, stopThr=5e-2,
    #                               numIterMax=1000, xtrain=x_train, xtest=x_test,
    #                               precomputed_train_train_wassertein_distances = precomputed_train_train_wassertein_distances,
    #                               precomputed_train_test_wassertein_distances = precomputed_train_test_wassertein_distances,
    #                                           precomputed_test_test_wassertein_distances = precomputed_test_test_wassertein_distances,y_train = y_train,Maxiters_for_variance=50),
    #                           likelihood=gpflow.likelihoods.MultiClass(num_classes=C,
    #                                                                    invlink=gpflow.likelihoods.RobustMax(
    #                                                                        num_classes=C)),
    #                           num_latent_gps=C)
    
        
    # set trainable parameters
    # gpflow.utilities.set_trainable(model.kernel.inverse_variance, True)
    # gpflow.utilities.set_trainable(model.kernel.scale, True)
    # gpflow.utilities.set_trainable(model.likelihood.invlink.epsilon, False)
    # model.kernel.mode = "training"

    # # see model parameters after training
    # gpflow.utilities.print_summary(model)

    # # running optimization
    # opt = gpflow.optimizers.Scipy()
    # opt_logs = opt.minimize(
    #     model.training_loss_closure(compile=False),
    #     model.trainable_variables,
    #     compile=False
    # )

    # # see model parameters after training
    # gpflow.utilities.print_summary(model)
    model = gpflow_new_kernels_modifed_2_Inverse_variance.get_optimal_inverse_variance(precomputed_train_train_wassertein_distances,
                                                                                       precomputed_train_test_wassertein_distances, 
                                                                                       precomputed_test_test_wassertein_distances,
                                                                                       x_train, x_test, y_train, y_test, Maxiters = 60)
    

    # Make Predictions
    model.kernel.mode = "testing"
    x_test_mean_predicted, x_test_var_predicted = model.predict_f(x_test)
    y_predicted = tf.argmax(x_test_mean_predicted, axis=1)
    print("y_predicted: ",'\n',y_predicted[:10])
    print("y_test: ",'\n',y_test[:10])
    print("Classification Accuracy: " + str(accuracy_score(y_true=y_test, y_pred=y_predicted)))
    all_accuracies.append(accuracy_score(y_true=y_test, y_pred=y_predicted))
  
    #2

# print("Num test per class = ", num_test_per_class)
# print("Num Train Per Class = ", num_training_per_class)
end = time.time()
print(end - start)  
print("All accuracies = ", all_accuracies)