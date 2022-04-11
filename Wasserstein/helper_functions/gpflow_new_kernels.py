import gpflow
import numpy as np
import tensorflow as tf
from gpflow.utilities import positive
from helper_functions import helpers
import parmap


class EWNonPsd(gpflow.kernels.Kernel):
    def __init__(self, cost_matrix: np.ndarray, xtrain: np.ndarray, xtest: np.ndarray,
                 initial_variance=1.0, initial_scale=1.0, reg=0.4, stopThr=1e-9, numIterMax=1000):
        super().__init__()
        self.variance = gpflow.Parameter(initial_variance, transform=positive())
        self.scale = gpflow.Parameter(initial_scale, transform=positive())
        self.cost_matrix = cost_matrix
        self.reg = reg
        self.stopThr = stopThr
        self.numIterMax = numIterMax

        # xtrain and xtest are needed to save all Wassertein distances in advance
        # and avoid re-computation on each iteration
        self.all_wassertein_distances = dict()
        print("Pre-computing Wassertein Distances .... ")
        train_train_wassertein_distances = helpers.get_wassertein_distances(cost_matrix=cost_matrix,
                                                                            x1=xtrain, x2=None,
                                                                            maxIters=self.numIterMax,
                                                                            stop_threshold=self.stopThr)
        for i in range(xtrain.shape[0]):
            for j in range(xtrain.shape[0]):
                self.all_wassertein_distances[(hash(xtrain[i].tostring()),
                                               hash(xtrain[j].tostring()))] = train_train_wassertein_distances[i, j]
                self.all_wassertein_distances[(hash(xtrain[j].tostring()),
                                               hash(xtrain[i].tostring()))] = train_train_wassertein_distances[i, j]
        test_train_wassertein_distances = helpers.get_wassertein_distances(cost_matrix=cost_matrix,
                                                                           x1=xtest, x2=xtrain,
                                                                           maxIters=self.numIterMax,
                                                                           stop_threshold=self.stopThr)
        for i in range(xtest.shape[0]):
            for j in range(xtrain.shape[0]):
                self.all_wassertein_distances[(hash(xtest[i].tostring()),
                                               hash(xtrain[j].tostring()))] = test_train_wassertein_distances[i, j]
                self.all_wassertein_distances[(hash(xtrain[j].tostring()),
                                               hash(xtest[i].tostring()))] = test_train_wassertein_distances[i, j]
        test_test_wassertein_distances = helpers.get_wassertein_distances(cost_matrix=cost_matrix,
                                                                          x1=xtest, x2=None,
                                                                          maxIters=self.numIterMax,
                                                                          stop_threshold=self.stopThr)
        for i in range(xtest.shape[0]):
            for j in range(xtest.shape[0]):
                self.all_wassertein_distances[(hash(xtest[i].tostring()),
                                               hash(xtest[j].tostring()))] = test_test_wassertein_distances[i, j]
                self.all_wassertein_distances[(hash(xtest[j].tostring()),
                                               hash(xtest[i].tostring()))] = test_test_wassertein_distances[i, j]
        print("Pre-computation of Wassertein Distances Completed ")

    def K(self, X, X2=None):
        if not tf.executing_eagerly():
            print("Error: Not Executing Eagerly, Please disable all calls to @tf.function.")
            exit(-1)
        if X2 is None:
            try:
                x1 = X.numpy()
            except AttributeError:
                x1 = np.array(X)
            wassertein_distance_matrix = np.zeros((x1.shape[0], x1.shape[0]))
            for i in range(x1.shape[0]):
                for j in range(x1.shape[0]):
                    wassertein_distance_matrix[i, j] = self.all_wassertein_distances[(hash(x1[i].tostring()),
                                                                                      hash(x1[j].tostring()))]
        else:
            try:
                x1 = X.numpy()
                x2 = X2.numpy()
            except AttributeError:
                x1 = np.array(X)
                x2 = np.array(X2)
            wassertein_distance_matrix = np.zeros((x1.shape[0], x2.shape[0]))
            for i in range(x1.shape[0]):
                for j in range(x2.shape[0]):
                    wassertein_distance_matrix[i, j] = self.all_wassertein_distances[(hash(x1[i].tostring()),
                                                                                      hash(x2[j].tostring()))]
        wassertein_distance_matrix = tf.constant(wassertein_distance_matrix, dtype=tf.dtypes.float64)

        # print("kernel: ")        
        # print(self.scale*tf.exp(-0.5*tf.divide(wassertein_distance_matrix, self.variance)))
        return self.scale*tf.exp(-0.5*tf.divide(wassertein_distance_matrix, self.variance))

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))


class ExpWassWithPsdApproximation(gpflow.kernels.Kernel):
    def __init__(self, cost_matrix: np.ndarray, xtrain: np.ndarray, xtest: np.ndarray,
                 initial_variance=1.0, initial_scale=1.0, reg=0.4, stopThr=1e-9, numIterMax=1000,
                 precomputed_test_test_wassertein_distances=None):
        super().__init__()
        self.variance = gpflow.Parameter(initial_variance, transform=positive())
        self.scale = gpflow.Parameter(initial_scale, transform=positive())
        self.cost_matrix = cost_matrix
        self.reg = reg
        self.stopThr = stopThr
        self.numIterMax = numIterMax
        self.eigen_vals = None
        self.eigen_vecs = None
        self.mode = "training"
        self.train_phi_x = None
        self.test_phi_x = None
        self.x_train = xtrain
        self.x_test = xtest

        print("Pre-computing Wassertein Distances .... ")
        self.train_train_wassertein_distances = helpers.get_wassertein_distances(cost_matrix=cost_matrix,
                                                                                 x1=xtrain, x2=None,
                                                                                 maxIters=self.numIterMax,
                                                                                 stop_threshold=self.stopThr)
        self.test_train_wassertein_distances = helpers.get_wassertein_distances(cost_matrix=cost_matrix,
                                                                                x1=xtest, x2=xtrain,
                                                                                maxIters=self.numIterMax,
                                                                                stop_threshold=self.stopThr)
        if precomputed_test_test_wassertein_distances is not None:
            self.test_test_wassertein_distances = precomputed_test_test_wassertein_distances
        else:
            self.test_test_wassertein_distances = helpers.get_wassertein_distances(cost_matrix=cost_matrix,
                                                                                   x1=xtest, x2=None,
                                                                                   maxIters=self.numIterMax,
                                                                                   stop_threshold=self.stopThr)
        print("Pre-computation of Wassertein Distances Completed ")

    def K(self, X, X2=None):
        if not tf.executing_eagerly():
            print("Error: Not Executing Eagerly, Please disable all calls to @tf.function.")
            exit(-1)
        if X2 is None:
            try:
                x1 = X.numpy()
            except AttributeError:
                x1 = np.array(X)
            if np.all(x1 == self.x_train):
                wassertein_distance_matrix = self.train_train_wassertein_distances
            elif np.all(x1 == self.x_test):
                wassertein_distance_matrix = self.test_test_wassertein_distances
            else:
                print("Asked for x1, None; with x1 != x_train or x_test")
                exit(-1)
        else:
            print("X, X2: ", X.shape, X2.shape)
            try:
                x1 = X.numpy()
                x2 = X2.numpy()
            except AttributeError:
                x1 = np.array(X)
                x2 = np.array(X2)
            if (x1 == self.x_train).all() and (x2 == self.x_test).all():
                wassertein_distance_matrix = self.test_train_wassertein_distances.T
            elif (x2 == self.x_train).all() and (x1 == self.x_test).all():
                wassertein_distance_matrix = self.test_train_wassertein_distances
            else:
                print("Asked for x1, None; with x1 != x_train or x_test")
                exit(-1)

        wassertein_distance_matrix = tf.constant(wassertein_distance_matrix, dtype=tf.dtypes.float64)
        exp_wassertein_non_psd_matrix = tf.exp(-0.5*tf.divide(wassertein_distance_matrix, self.variance))

        if self.mode == "training" or X2 is None:
            eigen_vals, eigen_vecs = tf.linalg.eigh(tensor=exp_wassertein_non_psd_matrix)
            if self.mode == "training":
                self.eigen_vals = eigen_vals
                self.eigen_vecs = eigen_vecs
                self.train_phi_x = self.create_phi_x_mat(exponential_wassertein_kernel=exp_wassertein_non_psd_matrix)
                # print("train_phi_x.shape", self.train_phi_x.shape)
            # eigen values are sorted in non decreasing order and corresponding eigen vectors are arranged in columns
            approximated_kernel = tf.zeros(shape=exp_wassertein_non_psd_matrix.shape, dtype=tf.dtypes.float64)
            for i in range(eigen_vals.shape[0]):
                if eigen_vals[i].numpy() > 0:
                    approximated_kernel += tf.multiply(eigen_vals[i], tf.matmul(eigen_vecs[:, i:i+1], eigen_vecs[:, i:i+1],
                                                                                transpose_a=False, transpose_b=True))
            return tf.multiply(self.scale, approximated_kernel)
        elif self.mode == "testing":
            if X2 is not None:
                # print("exp_wassertein_non_psd_matrix.shape: ", exp_wassertein_non_psd_matrix.shape)
                if exp_wassertein_non_psd_matrix.shape[1] == self.eigen_vals.shape[0]:
                    transposed = False
                else:
                    transposed = True
                    exp_wassertein_non_psd_matrix = tf.transpose(exp_wassertein_non_psd_matrix)
                test_phi_x = self.create_phi_x_mat(exponential_wassertein_kernel=exp_wassertein_non_psd_matrix)
                # print("Shapes: test_phi_x, train_phi_x", test_phi_x.shape, self.train_phi_x.shape)
                approximated_kernel = tf.matmul(test_phi_x, tf.transpose(self.train_phi_x))
                if not transposed:
                    return approximated_kernel
                else:
                    return tf.multiply(self.scale, tf.transpose(approximated_kernel))
            else:
                print("Asking for testing only: ", X.shape)
                exit(-1)
        else:
            print("Error: Wrong Mode Specified")
            exit(-1)

    def create_phi_x_mat(self, exponential_wassertein_kernel):
        L = np.sum(self.eigen_vals.numpy() > 0)
        # eigvals is (L,) shaped array of eigenvalues, corresponding eigenvectors are arranged column wise in eigvecs
        eigvals = self.eigen_vals[-L:]
        eigvecs = self.eigen_vecs[:, -L:]
        phi_x_mat = tf.zeros((exponential_wassertein_kernel.shape[0], L))
        phi_x_rows_list = list()
        for i in range(exponential_wassertein_kernel.shape[0]):
            kx = exponential_wassertein_kernel[i:i+1, :]
            # print("ew_k_n_psd, kx, eigvecs, eigvals, mode", exponential_wassertein_kernel.shape, kx.shape, eigvecs.shape, eigvals.shape, self.mode)
            phi_x_row = tf.divide(tf.matmul(kx, eigvecs, transpose_a=False, transpose_b=False), tf.math.sqrt(eigvals))
            # print(phi_x_row[0].shape)
            phi_x_rows_list.append(phi_x_row[0])
        phi_x_mat = tf.stack(phi_x_rows_list)
        return phi_x_mat

    def K_diag(self, X):
        return tf.linalg.diag_part(self.K(X))


# x_train, y_train, x_test, y_test = helpers.load_mnist_data(num_train_samples=6, num_test_samples=7)
# x_train, x_test = helpers.map_to_probability_simplex([x_train, x_test])
# m = helpers.get_image_cost_matrix(28, 28)
# C = 10
# model = gpflow.models.VGP(data=(x_train, y_train),
#                           mean_function=gpflow.mean_functions.Zero(),
#                           kernel=EWNonPsd(cost_matrix=m, initial_variance=1.0),
#                           likelihood=gpflow.likelihoods.MultiClass(num_classes=C,
#                                                                    invlink=gpflow.likelihoods.RobustMax(num_classes=C)),
#                           num_latent_gps=C)
#
# # set trainable parameters
# gpflow.utilities.set_trainable(model.kernel.variance, True)
# gpflow.utilities.set_trainable(model.likelihood.invlink.epsilon, False)
# gpflow.utilities.print_summary(model)
#
# # running optimization
# opt = gpflow.optimizers.Scipy()
# opt_logs = opt.minimize(
#     model.training_loss_closure(compile=False),
#     model.trainable_variables,
#     compile=False
# )
#
# # see model parameters after training
# gpflow.utilities.print_summary(model)
#
# # Make Predictions
# x_test_mean_predicted, x_test_var_predicted = model.predict_f(x_test)
# y_predicted = tf.argmax(x_test_mean_predicted, axis=1)


# class PowerScaledLinearOnPsdEwFeatureSpace(gpflow.kernels.Linear):
#     """
#     The kernel equation is
#         k(x, y) = (σ²xy + offset)ᵈ
#     where:
#     σ² is the variance parameter,
#     offset is the offset parameter,
#     d is the degree parameter.
#     """
#
#     def __init__(self, degree=3.0, variance=1.0, offset=0.0, active_dims=None):
#         """
#         :param degree: the degree of the polynomial
#         :param variance: the (initial) value for the variance parameter(s),
#             to induce ARD behaviour this must be initialised as an array the same
#             length as the the number of active dimensions e.g. [1., 1., 1.]
#         :param offset: the offset of the polynomial
#         :param active_dims: a slice or list specifying which columns of X are used
#         """
#         super().__init__(variance, active_dims)
#         self.degree = gpflow.base.Parameter(degree, transform=positive())
#         self.offset = offset
#
#     def K(self, X, X2=None):
#         return (super().K(X, X2) + self.offset) ** self.degree
#
#     def K_diag(self, X):
#         return (super().K_diag(X) + self.offset) ** self.degree

# class PowerScaledLinearOnPsdEwFeatureSpace(gpflow.kernels.Kernel):
#     def __init__(self):
#         super().__init__()
#         self.power = gpflow.Parameter(1.0, transform=positive())
#         self.scale = gpflow.Parameter(1.0, transform=positive())
#
#     def K(self, X, X2=None):
#         # K(x,y) = (scale * (x.T@y)) ^power
#         if X2 is None:
#             X2 = tf.identity(X)
#         # print("ASKED FOR X:", X.shape, " X2:", X2.shape)
#         return tf.math.multiply(self.scale,
#                                 tf.math.pow(tf.linalg.matmul(a=X, b=X2, transpose_a=False, transpose_b=True), self.theta))
#
#     def K_diag(self, X):
#         return tf.linalg.diag_part(self.K(X))


