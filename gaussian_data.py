import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

def generate_gaussian_data(d, K=None, n=None, epsilon=None):
    # Set default values if not provided
    if K is None:
        K = d  # Default number of classes K is equal to d
    if n is None:
        n = 100  # Default number of samples per class if not provided
    if epsilon is None:
        epsilon = 0.1  # Default noise level if not provided

    Kn = K * n  # Total number of samples

    # Initialize the sample matrix X and the label matrix Y
    X_train = np.zeros((Kn, d))  # Shape (Kn, d)
    Y_train = np.zeros((Kn,))  # Shape (Kn)

    X_test = np.zeros((Kn, d))  # Shape (Kn, d)
    Y_test = np.zeros((Kn,))  # Shape (Kn)

    for k in range(K):
        # Generate n samples for class k
        mean = np.eye(d)[k]  # e_k: the standard unit vector for class k
        cov = (epsilon ** 2) * np.eye(d)  # Covariance matrix: epsilon^2 * I_d

        # Generate n samples from N(e_k, epsilon^2 I_d)
        samples = np.random.multivariate_normal(mean, cov, n)

        # Assign the generated samples to the corresponding block in X
        X_train[k * n: (k + 1) * n, :] = samples
        # Set the corresponding labels in Y to one-hot vectors for class k
        Y_train[k * n: (k + 1) * n] = k

        # Assign the generated samples to the corresponding block in X
        X_test[k * n: (k + 1) * n, :] = samples
        # Set the corresponding labels in Y to one-hot vectors for class k
        Y_test[k * n: (k + 1) * n] = k

    # Convert data to PyTorch tensors
    X_train = torch.tensor(X_train, dtype=torch.float32)
    Y_train = torch.tensor(Y_train, dtype=torch.int64)

    # Create a DataLoader for batch processing
    train_dataset = TensorDataset(X_train, Y_train)

    # Convert data to PyTorch tensors
    X_test = torch.tensor(X_test, dtype=torch.float32)
    Y_test = torch.tensor(Y_test, dtype=torch.int64)

    # Create a DataLoader for batch processing
    test_dataset = TensorDataset(X_test, Y_test)

    return train_dataset, test_dataset