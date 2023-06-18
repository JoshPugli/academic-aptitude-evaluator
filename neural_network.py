from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt

PLOT = False
TRAIN = True


def load_data(base_path="../proj/data"):
    """ Load the data in PyTorch Tensor.

    :return: (zero_train_matrix, train_data, valid_data, test_data)
        WHERE:
        zero_train_matrix: 2D sparse matrix where missing entries are
        filled with 0.
        train_data: 2D sparse matrix
        valid_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
        test_data: A dictionary {user_id: list,
        user_id: list, is_correct: list}
    """
    train_matrix = load_train_sparse(base_path).toarray()
    valid_data = load_valid_csv(base_path)
    test_data = load_public_test_csv(base_path)

    zero_train_matrix = train_matrix.copy()
    # Fill in the missing entries to 0.
    zero_train_matrix[np.isnan(train_matrix)] = 0
    # Change to Float Tensor for PyTorch.
    zero_train_matrix = torch.FloatTensor(zero_train_matrix)
    train_matrix = torch.FloatTensor(train_matrix)

    return zero_train_matrix, train_matrix, valid_data, test_data


class AutoEncoder(nn.Module):
    def __init__(self, num_question, k=100):
        """ Initialize a class AutoEncoder.

        :param num_question: int
        :param k: int
        """
        super(AutoEncoder, self).__init__()

        # Define linear functions.
        self.g = nn.Linear(num_question, k)
        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight.clone().detach(), 2) ** 2
        h_w_norm = torch.norm(self.h.weight.clone().detach(), 2) ** 2
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        # relu vs simgoid try for optim.
        x = torch.sigmoid(self.g(inputs))
        x = torch.sigmoid(self.h(x))
        out = x
        return out


def train(model, lr, lamb, train_data, zero_train_data, valid_data, num_epoch):
    """ Train the neural network, where the objective also includes
    a regularizer.

    :param model: Module
    :param lr: float
    :param lamb: float
    :param train_data: 2D FloatTensor
    :param zero_train_data: 2D FloatTensor
    :param valid_data: Dict
    :param num_epoch: int
    :return: None
    """
    # Add a regularizer to the cost function.
    # weight_norm = (lamb / 2) * model.get_weight_norm()

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr)
    # optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=lamb)

    num_student = train_data.shape[0]

    train_cost = []
    valid_accuracies = []

    for epoch in range(0, num_epoch):
        train_loss = 0.

        for user_id in range(num_student):
            inputs = Variable(zero_train_data[user_id]).unsqueeze(0)
            target = inputs.clone()

            optimizer.zero_grad()
            output = model(inputs)

            # Mask the target to only compute the gradient of valid entries.
            nan_mask = np.isnan(train_data[user_id].unsqueeze(0).numpy())
            target[0][nan_mask] = output[0][nan_mask]

            # loss = torch.sum((output - target) ** 2) + weight_norm  # 0.68
            weight_norm = (lamb/2) * model.get_weight_norm()
            loss = torch.sum((output - target) ** 2) + weight_norm
            loss.backward(retain_graph=True)

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_accuracies.append(valid_acc)
        train_cost.append(train_loss)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    return valid_accuracies, train_cost


def evaluate(model, train_data, valid_data):
    """ Evaluate the valid_data on the current model.

    :param model: Module
    :param train_data: 2D FloatTensor
    :param valid_data: A dictionary {user_id: list,
    question_id: list, is_correct: list}
    :return: float
    """
    # Tell PyTorch you are evaluating the model.
    model.eval()

    total = 0
    correct = 0

    for i, u in enumerate(valid_data["user_id"]):
        inputs = Variable(train_data[u]).unsqueeze(0)
        output = model(inputs)

        guess = output[0][valid_data["question_id"][i]].item() >= 0.5
        if guess == valid_data["is_correct"][i]:
            correct += 1
        total += 1
    return correct / float(total)


def main():
    zero_train_matrix, train_matrix, valid_data, test_data = load_data()

    #####################################################################
    # Try out 5 different k and select the best k using the             #
    # validation set.                                                   #
    #####################################################################
    # Set model hyperparameters.
    max_acc = 0
    best_k = 200
    best_lr = 0.1
    best_lamb = 0.00001
    best_lamb = 0
    best_epoch = 10

    if TRAIN:
        ks = [10, 50, 100, 200, 500]

        # Set optimization hyperparameters.
        lrs = [2, 1, 0.1, 0.01, 0.001, 0.0001]
        lambs = [0.001, 0.01, 0.1, 1]

        for k in ks:
            for lr in lrs:
                for lamb in lambs:

                    model = AutoEncoder(train_matrix.shape[1], best_k)
                    acc = train(model, best_lr, lamb, train_matrix,
                                zero_train_matrix, valid_data, best_epoch)[0]
                    if max(acc) > max_acc:
                        max_acc = max(acc)
                        best_k = k
                        best_lr = lr
                        best_lamb = lamb

                    print("Best k: {} \tBest lr: {} \tBest lamb: {} \tBest Acc: {}".format(
                        best_k, best_lr, best_lamb, max_acc))
                    
        print("FINAL: Best k: {} \tBest lr: {} \tBest lamb: {} \tBest Acc: {}".format(
            best_k, best_lr, best_lamb, max_acc
        ))

    # plot and report the how the training and validation objectives
    # change as a function of the number of epochs.

    if PLOT:

        model = AutoEncoder(train_matrix.shape[1], best_k)
        valid_accs, train_cost = train(model, best_lr, best_lamb, train_matrix,
                                       zero_train_matrix, valid_data, best_epoch)

        fig = plt.figure()
        plt.title("Validation Accuracy vs. Epoch")
        plt.plot(range(best_epoch), valid_accs)
        plt.scatter(range(best_epoch), valid_accs, label="Validation Accuracy")
        plt.xlabel("Epoch")
        plt.ylabel("Training Cost")
        plt.legend()
        plt.savefig("train_cost.png")

        fig = plt.figure()
        plt.title("Training Cost vs. Epoch")
        plt.plot(range(best_epoch), train_cost)
        plt.scatter(range(best_epoch), train_cost, label="Training Cost")
        plt.xlabel("Epoch")
        plt.ylabel("Training Cost")
        plt.legend()
        plt.savefig("valid_acc.png")

    test_accs, _ = train(model, best_lr, best_lamb, train_matrix,
                         zero_train_matrix, test_data, best_epoch)
    print("Test Accuracy: {}".format(test_accs[-1]))


if __name__ == "__main__":
    main()