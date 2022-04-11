import numpy as np
from mnist.loader import MNIST


def data_loader(dataset: str, dataset_type: str, num_each_class_train: int = 1, num_each_class_test: int = 1, seed=42):
    np.random.seed(42)

    if dataset.strip().lower() == "mnist":
        mnist = MNIST('data/MNIST')
        x_train, y_train = mnist.load_training()
        x_test, y_test = mnist.load_testing()
        x_train, y_train, x_test, y_test = list(map(lambda x: np.array(x).astype(np.float64),
                                                    [x_train, y_train, x_test, y_test]))

        if dataset_type == "full":
            return x_train, y_train, x_test, y_test
        else:
            classes = np.unique(y_train)
            x_train_filtered, y_train_filtered, x_test_filtered, y_test_filtered = [None]*4
            for label in classes:
                train_indices = np.random.choice(np.where(y_train == label)[0], size=num_each_class_train, replace=False)
                test_indices = np.random.choice(np.where(y_test == label)[0], size=num_each_class_test, replace=False)
                if x_train_filtered is None:
                    x_train_filtered, y_train_filtered = x_train[train_indices], y_train[train_indices]
                    x_test_filtered, y_test_filtered = x_test[test_indices], y_test[test_indices]
                else:
                    x_train_filtered = np.concatenate([x_train_filtered, x_train[train_indices]])
                    y_train_filtered = np.concatenate([y_train_filtered, y_train[train_indices]])
                    x_test_filtered = np.concatenate([x_test_filtered, x_test[test_indices]])
                    y_test_filtered = np.concatenate([y_test_filtered, y_test[test_indices]])

            indices_train = np.arange(x_train_filtered.shape[0])
            indices_test = np.arange(x_test_filtered.shape[0])
            np.random.shuffle(indices_train)
            np.random.shuffle(indices_test)
            x_train_filtered, y_train_filtered = x_train_filtered[indices_train], y_train_filtered[indices_train]
            x_test_filtered, y_test_filtered = x_test_filtered[indices_test], y_test_filtered[indices_test]

            return x_train_filtered, y_train_filtered, x_test_filtered, y_test_filtered


# res = data_loader('MNIST', 'partial', 16, 10)
# # print(len(res))
# for x in res:
#     print(x.shape, end=' ')
# print()
