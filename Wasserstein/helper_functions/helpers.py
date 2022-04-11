import numpy as np
import ot
from tqdm import tqdm
# from mnist import MNIST
from mnist.loader import MNIST
import sklearn
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")
np.random.seed(42)


def load_mnist_data(num_train_samples, num_test_samples):
    mnist = MNIST('data/MNIST')
    x_train, y_train = mnist.load_training()
    x_test, y_test = mnist.load_testing()
    x_train, y_train, x_test, y_test = map(lambda x: np.array(x).astype(np.float32), [x_train, y_train, x_test, y_test])
    idx_train = np.random.choice(np.arange(x_train.shape[0]), size=num_train_samples, replace=False)
    idx_test = np.random.choice(np.arange(x_test.shape[0]), size=num_test_samples, replace=False)
    x_train, y_train = x_train[idx_train], y_train[idx_train]
    x_test, y_test = x_test[idx_test], y_test[idx_test]
    # print("Loaded Data:",  x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    return x_train, y_train, x_test, y_test


def get_image_cost_matrix(nsx, nsy):
    ix = np.zeros((nsx, nsy))
    nt = np.int32(nsx * nsy)
    for i in range(nsx):
        ix[i, :] = np.arange(nsy)
    ix = ix.flatten()
    iy = np.zeros((nsx, nsy))
    for i in range(nsy):
        iy[:, i] = np.arange(nsx)
    iy = iy.flatten()
    # creating the cost matrix
    m = np.zeros((nt, nt))
    for i in range(nt):
        for j in range(nt):
            m[i, j] = (ix[i] - ix[j])**2 + (iy[i] - iy[j])**2
    np.fill_diagonal(m, 0)
    return m


def spectral_decomposition(real_symmetric_matrix, min_eigenvalue=10**-6):
    eigvals, eigvecs = np.linalg.eigh(real_symmetric_matrix)
    idx_eigvals = np.argsort(eigvals)
    eigvals, eigvecs = eigvals[idx_eigvals], eigvecs[:, idx_eigvals]
    # print(eigvals)
    idx_eigvals = eigvals > min_eigenvalue  # take eigen vectors with eigenvalues > 10^-6
    eigvals, eigvecs = eigvals[idx_eigvals], eigvecs[:, idx_eigvals]
    L = len(eigvals)
    return L, eigvals, eigvecs


def get_wassertein_distances(cost_matrix, x1, x2=None, epsilon=0.4, maxIters=1000,
                             stop_threshold: float = 5e-2, label='Creating Wassertein Distance Matrix'):
    if x2 is None:
        wassertein_dists = np.zeros((x1.shape[0], x1.shape[0]))
        for i in tqdm(range(x1.shape[0]), label):
            wassertein_dists[i, :] = ot.sinkhorn2(a=x1[i], b=x1.T, M=cost_matrix,
                                                  reg=epsilon, numItermax=maxIters, stopThr=stop_threshold)
        np.fill_diagonal(wassertein_dists, 0)
    else:
        wassertein_dists = np.zeros((x1.shape[0], x2.shape[0]))
        for i in tqdm(range(x1.shape[0]), 'Creating Wassertein Distances for Test Set'):
            wassertein_dists[i, :] = ot.sinkhorn2(a=x1[i], b=x2.T, M=cost_matrix,
                                                  reg=epsilon, numItermax=maxIters, stopThr=stop_threshold)
    return wassertein_dists


def map_to_probability_simplex(x: list or np.array, jitter=True, jitter_val: float = 1e-6):
    if type(x) == np.ndarray:
        assert len(x.shape) == 2
        for i in range(x.shape[0]):
            if jitter:
                x[i] += jitter_val
            x[i] /= np.sum(x[i])
        return x
    elif type(x) == list:
        result, array_list = [], x.copy()
        for x in array_list:
            assert len(x.shape) == 2
            for i in range(x.shape[0]):
                if jitter:
                    x[i] += jitter_val
                x[i] /= np.sum(x[i])
            result.append(x)
        return result


def map_to_exponential_wassertein_feature_space(exp_wassertein_kernel: np.ndarray, eigvals, eigvecs) -> np.ndarray:
    # eigvals is (L,) shaped array of eigenvalues, corresponding eigenvectors are arranged column wise in eigvecs
    # exp_wassertein_kernel.shape = (n_train, n_train) if creating feature vector for training set
    #                               (n_val, n_train) if creating feature vector for validation set
    #                               (n_test, n_train) if creating feature vector for testing set
    assert len(exp_wassertein_kernel.shape) == 2 and len(eigvecs.shape) == 2 and len(eigvals) == eigvecs.shape[1]
    L = len(eigvals)
    phi_x_mat = np.zeros((exp_wassertein_kernel.shape[0], L))
    for i in range(exp_wassertein_kernel.shape[0]):
        kx = exp_wassertein_kernel[i, :]
        phi_x = np.reciprocal(np.sqrt(eigvals)) * np.array(
            list(map(lambda v_i: np.dot(kx, v_i), [eigvecs[:, idx] for idx in range(L)])))
        phi_x_mat[i, :] = phi_x
    return phi_x_mat


def create_exponential_wassertein_kernel(wassertein_dists: np.ndarray or list,
                                         sigma: str = 'unknown') -> (float, np.ndarray or list) or (np.ndarray or list):
    if type(wassertein_dists) == np.ndarray:
        if sigma == 'unknown':
            sigma = np.std(wassertein_dists.flatten(), ddof=1)
            wassertein_kernel = np.exp(-(wassertein_dists/sigma)**2)
            return sigma, wassertein_kernel
        else:
            wassertein_kernel = np.exp(-(wassertein_dists / sigma) ** 2)
            return wassertein_kernel
    else:
        wassertein_dists_list, exp_wassertein_kernels = wassertein_dists.copy(), list()
        if sigma == 'unknown':
            sigma = np.std(wassertein_dists[0].flatten(), ddof=1)
            for wassertein_dists in wassertein_dists_list:
                exp_wassertein_kernels.append(np.exp(-(wassertein_dists/sigma)**2))
            return sigma, exp_wassertein_kernels
        else:
            for wassertein_dists in wassertein_dists_list:
                exp_wassertein_kernels.append(np.exp(-(wassertein_dists / sigma) ** 2))
            return exp_wassertein_kernels


def create_exponential_wassertein_feature_vector(x_train, x_test, epsilon=0.4, stop_threshold=5e-2, normalize=False):

    nsx, nsy = np.int32(np.sqrt(x_train.shape[1])), np.int32(np.sqrt(x_train.shape[1]))

    # create cost matrix
    m = get_image_cost_matrix(nsx, nsy)

    # map data to probability simplex
    x_train, x_test = map_to_probability_simplex([x_train, x_test], jitter=True, jitter_val=1e-6)

    # creating wassertein distance matrix
    wassertein_dists = get_wassertein_distances(cost_matrix=m, x1=x_train, x2=None,
                                                epsilon=epsilon, maxIters=1000, stop_threshold=stop_threshold,
                                                label='Creating Wassertein Distance Matrix (Train Set)')
    wassertein_dists_test = get_wassertein_distances(cost_matrix=m, x1=x_test, x2=x_train,
                                                     epsilon=epsilon, maxIters=1000, stop_threshold=stop_threshold,
                                                     label='Creating Wassertein Distance Matrix (Test Set)')

    # creating exponential wassertein kernel matrix
    sigma, k_W = create_exponential_wassertein_kernel(wassertein_dists=wassertein_dists, sigma='unknown')
    k_W_test = create_exponential_wassertein_kernel(wassertein_dists=wassertein_dists_test, sigma=sigma)

    # spectral decomposition
    L, eigvals, eigvecs = spectral_decomposition(real_symmetric_matrix=k_W, min_eigenvalue=10 ** (-6))

    # Creating Feature Vectors for both test and train
    phi_x_mat_train = map_to_exponential_wassertein_feature_space(exp_wassertein_kernel=k_W,
                                                                  eigvals=eigvals, eigvecs=eigvecs)
    phi_x_mat_test = map_to_exponential_wassertein_feature_space(exp_wassertein_kernel=k_W_test,
                                                                 eigvals=eigvals, eigvecs=eigvecs)

    if normalize:
        # Normalizing the Data after mapping to Exponential Wassertein Space
        scaler = StandardScaler()
        scaler.fit(phi_x_mat_train)
        phi_x_mat_train, phi_x_mat_test = scaler.transform(phi_x_mat_train), scaler.transform(phi_x_mat_test)

    return phi_x_mat_train, phi_x_mat_test


def get_classification_score(clf, x_train, y_train, x_test, y_test, normalize_before_fitting=False):
    if normalize_before_fitting:
        scaler = StandardScaler()
        scaler.fit(x_train)
        x_train, x_test = scaler.transform(x_train), scaler.transform(x_test)
    y_pred = clf.fit(x_train, y_train).predict(x_test)
    score = accuracy_score(y_pred=y_pred, y_true=y_test)
    return score