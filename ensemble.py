import numpy as np
from utils import *
from item_response import neg_log_likelihood, update_theta_beta, sigmoid
from sklearn.impute import KNNImputer
from knn import knn_impute_by_user
import random

def knn_train(matrix, k):
    nbrs = KNNImputer(n_neighbors=k)
    # We use NaN-Euclidean distance measure.
    mat = nbrs.fit_transform(matrix)
    return mat


def irt_train(C, Cval, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta)
    """
    # TODO: Initialize theta and beta.
    theta = np.zeros(C.shape[0])
    beta = np.zeros(C.shape[1])

    trainLLK = []
    valLLK = []

    for _ in range(iterations):
        neg_lld = neg_log_likelihood(C, theta=theta, beta=beta)
        trainLLK.append(-neg_lld)
        valLLK.append(-neg_log_likelihood(Cval, theta=theta, beta=beta))
        theta, beta = update_theta_beta(C, lr, theta, beta)

    return theta, beta


def generate_matrix(sparse_matrix):
    mask = np.isnan(sparse_matrix)

    non_nan_locs = np.argwhere(~mask)
    hidden_num = int(non_nan_locs.shape[0] * 0.1)
    sample_indeces = np.random.choice(list(range(len(non_nan_locs))), hidden_num, replace=True)
    hidden_indeces = non_nan_locs[sample_indeces]
    sample = np.copy(sparse_matrix)
    for sample_index in hidden_indeces:
        sample[sample_index[0]][sample_index[1]] = np.nan
    
    return sample


def ensemble_matrix_create(knn_matrix_1, knn_matrix_2, theta, beta):
    new_matrix = np.full((542, 1774), np.nan)
    
    for i in range(knn_matrix_1.shape[0]):
        for j in range(knn_matrix_1.shape[1]):
            
            prob_1 = knn_matrix_1[i][j]
            prob_2 = knn_matrix_2[i][j]
            prob_3 = sigmoid(theta[i] - beta[j])
            
            prob_avg = (prob_1 + prob_2 + prob_3) / 3
            
            new_matrix[i][j] = prob_avg
            
    return new_matrix
            

def main():
    sparse_matrix = load_train_sparse("../311-project/data").toarray()
    val_data = load_valid_csv("../311-project/data")
    test_data = load_public_test_csv("../311-project/data")
    
    matrix_1 = knn_train(generate_matrix(sparse_matrix), 11)
    matrix_2 = knn_train(generate_matrix(sparse_matrix), 11)
    # matrix_3 = knn_train(generate_matrix(sparse_matrix), 11)
    
    irt_dataset = generate_matrix(sparse_matrix)
    Cval = np.empty_like(irt_dataset)
    lr = 0.1
    iters = 100
    
    theta, beta = irt_train(irt_dataset, Cval, lr, iters)
    
    prob_matrix = ensemble_matrix_create(matrix_1, matrix_2, theta, beta)
    val_acc = sparse_matrix_evaluate(val_data, prob_matrix, threshold=0.5)
    print(val_acc)
    
    test_acc = sparse_matrix_evaluate(test_data, prob_matrix, threshold=0.5)
    print(test_acc)
    
if __name__ == "__main__":
    main()