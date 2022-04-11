from data_loader import loader
import pandas as pd
from helper_functions import helpers
from sklearn.preprocessing import StandardScaler
import numpy as np
np.random.seed(42)


def pre_process_data(xtrain, ytrain, xtest, ytest,
                     mapToProbability=False, standardScale=False, jitter=False, asFloat64=False):
    np.random.seed(42)
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


x_train, y_train, x_test, y_test = loader.data_loader(dataset="mnist", dataset_type='full', seed=42)
x_train, y_train, x_test, y_test = x_train[:5], y_train[:5], x_test[:3], y_test[:3]
x_train, y_train, x_test, y_test = pre_process_data(x_train, y_train, x_test, y_test,
                                                    mapToProbability=True, standardScale=False,
                                                    jitter=True, asFloat64=True)

mnist_distances = list()
cost_matrix = helpers.get_image_cost_matrix(nsx=28, nsy=28)

# train-train distances
train_train_dists = helpers.get_wassertein_distances(cost_matrix=cost_matrix, x1=x_train, x2=None)
for i in range(x_train.shape[0]):
    for j in range(x_train.shape[0]):
        mnist_distances.append({"array1": hash(x_train[i].tostring()), "array2": hash(x_train[j].tostring()),
                                "wassertein_distance": train_train_dists[i, j]})
        mnist_distances.append({"array1": hash(x_train[j].tostring()), "array2": hash(x_train[i].tostring()),
                                "wassertein_distance": train_train_dists[i, j]})

# test-train distances
test_train_dists = helpers.get_wassertein_distances(cost_matrix=cost_matrix, x1=x_test, x2=x_train)
for i in range(x_test.shape[0]):
    for j in range(x_train.shape[0]):
        mnist_distances.append({"array1": hash(x_test[i].tostring()), "array2": hash(x_train[j].tostring()),
                                "wassertein_distance": test_train_dists[i, j]})
        mnist_distances.append({"array1": hash(x_train[j].tostring()), "array2": hash(x_test[i].tostring()),
                                "wassertein_distance": test_train_dists[i, j]})

# test-test distances
test_test_dists = helpers.get_wassertein_distances(cost_matrix=cost_matrix, x1=x_test, x2=None)
for i in range(x_test.shape[0]):
    for j in range(x_test.shape[0]):
        mnist_distances.append({"array1": hash(x_test[i].tostring()), "array2": hash(x_test[j].tostring()),
                                "wassertein_distance": test_test_dists[i, j]})
        mnist_distances.append({"array1": hash(x_test[j].tostring()), "array2": hash(x_test[i].tostring()),
                                "wassertein_distance": test_test_dists[i, j]})

mnist_distances = pd.DataFrame(mnist_distances)
print(mnist_distances)
mnist_distances.to_csv('precomputed_wassertein_distances/mnist_wassertein_distances.csv')
