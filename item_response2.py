from utils import *

import numpy as np
import matplotlib.pyplot as plt


def sigmoid(x):
    """ Apply sigmoid function.
    """
    return np.exp(x) / (1 + np.exp(x))


def sigmoid2(x, c):
    return (1 - c) / (1 + np.exp(-x))


def neg_log_likelihood(C, theta, beta, alpha, gamma):
    """ Compute the negative log-likelihood.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    T = np.tile(theta[None,:].transpose(), (1, C.shape[1]))
    B = np.tile(beta, (C.shape[0], 1))
    A = np.tile(alpha, (C.shape[0], 1))
    G = np.tile(gamma, (C.shape[0], 1))
    log_lklihood = np.nansum(C * np.log(sigmoid2(A * (T - B), G) + G) \
        + (1 - C) * np.log(1 - (sigmoid2(A * (T - B), G) + G))) / np.count_nonzero(~np.isnan(C))

    return -log_lklihood


def update_theta_beta_alpha_gamma(C, lr, theta, beta, alpha, gamma):
    """ Update theta and beta using gradient descent.

    You are using alternating gradient descent. Your update should look:
    for i in iterations ...
        theta <- new_theta
        beta <- new_beta

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param theta: Vector
    :param beta: Vector
    :return: tuple of vectors
    """
    T = np.tile(theta[None,:].transpose(), (1, C.shape[1]))
    B = np.tile(beta, (C.shape[0], 1))
    A = np.tile(alpha, (C.shape[0], 1))
    G = np.tile(gamma, (C.shape[0], 1))

    Z = A * (T - B)
    Y = sigmoid2(Z, G) + G

    theta += lr * np.nansum((C / Y - (1 - C) / (1 - Y)) * sigmoid2(Z, G) * (1 - sigmoid(Z)) * A, axis=1) \
        / np.count_nonzero(~np.isnan(C), axis=1)
    beta += lr * np.nansum((C / Y - (1 - C) / (1 - Y)) * sigmoid2(Z, G) * (1 - sigmoid(Z)) * (-A), axis=0) \
        / np.count_nonzero(~np.isnan(C), axis=0)
    alpha += lr * np.nansum((C / Y - (1 - C) / (1 - Y)) * sigmoid2(Z, G) * (1 - sigmoid(Z)) * (T - B), axis=0) \
        / np.count_nonzero(~np.isnan(C), axis=0)
    gamma += lr * np.nansum((C / Y - (1 - C) / (1 - Y)) * (1 - sigmoid(Z)), axis=0) \
        / np.count_nonzero(~np.isnan(C), axis=0)
    
    alpha = np.maximum(0.001, alpha)
    gamma = np.maximum(0, gamma)

    return theta, beta, alpha, gamma


def irt(C, val_data, Cval, lr, iterations):
    """ Train IRT model.

    You may optionally replace the function arguments to receive a matrix.

    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param val_data: A dictionary {user_id: list, question_id: list,
    is_correct: list}
    :param lr: float
    :param iterations: int
    :return: (theta, beta, val_acc_lst)
    """
    theta = np.zeros(C.shape[0])
    beta = np.zeros(C.shape[1])
    alpha = np.ones(C.shape[1])
    gamma = np.zeros(C.shape[1])

    val_acc_lst = []
    trainLLK = []
    valLLK = []

    for i in range(iterations):
        neg_lld = neg_log_likelihood(C, theta=theta, beta=beta, alpha=alpha, gamma=gamma)
        trainLLK.append(-neg_lld)
        valLLK.append(-neg_log_likelihood(Cval, theta=theta, beta=beta, alpha=alpha, gamma=gamma))
        score = evaluate(data=val_data, theta=theta, beta=beta, alpha=alpha, gamma=gamma)
        val_acc_lst.append(score)
        print("NLLK: {} \t Score: {} \t Iter: {}".format(neg_lld, score, i))
        theta, beta, alpha, gamma = update_theta_beta_alpha_gamma(C, lr, theta, beta, alpha, gamma)

    return theta, beta, alpha, gamma, val_acc_lst, trainLLK, valLLK


def evaluate(data, theta, beta, alpha, gamma):
    """ Evaluate the model given data and return the accuracy.
    :param data: A dictionary {user_id: list, question_id: list,
    is_correct: list}

    :param theta: Vector
    :param beta: Vector
    :return: float
    """
    pred = []
    for i, q in enumerate(data["question_id"]):
        u = data["user_id"][i]
        x = (alpha[q] * (theta[u] - beta[q])).sum()
        p_a = sigmoid2(x, gamma[q]) + gamma[q]
        pred.append(p_a >= 0.5)
    return np.sum((data["is_correct"] == np.array(pred))) \
           / len(data["is_correct"])


def main():
    train_data = load_train_csv("../data")
    sparse_matrix = load_train_sparse("../data")
    val_data = load_valid_csv("../data")
    test_data = load_public_test_csv("../data")

    Cval = np.empty_like(sparse_matrix.A)
    Cval[:,:] = np.nan
    for i in range(len(val_data["user_id"])):
        Cval[val_data["user_id"][i],val_data["question_id"][i]] = val_data["is_correct"][i]

    lr = 0.1
    iters = 300
    theta, beta, alpha, gamma, val_acc_lst, trainLLK, valLLK = \
        irt(sparse_matrix.A, val_data, Cval, lr, iters)

    plt.plot(range(iters), valLLK, label="Valid LLK")
    plt.plot(range(iters), trainLLK, "-.", label="Train LLK")
    plt.xlabel("Iteration")
    plt.ylabel("LLK")
    plt.legend()
    plt.title("Training vs Validation LLK")
    plt.show()

    print(f"Final Validation Score: {evaluate(val_data, theta, beta, alpha, gamma)}")
    print(f"Final Testing Score: {evaluate(test_data, theta, beta, alpha, gamma)}")

    for j in [452, 1337, 311]:
        range_theta = np.linspace(-10, 10, num=500)
        long_beta = np.full(len(range_theta), beta[j])
        long_alpha = np.full(len(range_theta), alpha[j])
        long_gamma = np.full(len(range_theta), gamma[j])
        pjs = sigmoid2(long_alpha * (range_theta - long_beta), long_gamma) + long_gamma
        plt.plot(range_theta, pjs, label=f"Question {j}")
    plt.xlabel("Theta")
    plt.ylabel("Probability of Correct Response")
    plt.legend()
    plt.title("Probability of Correct Responses for 3 Questions")
    plt.show()


if __name__ == "__main__":
    main()
