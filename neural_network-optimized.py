from utils import *
from torch.autograd import Variable

import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data

import numpy as np
import torch

import matplotlib.pyplot as plt
from tsne_torch import TorchTSNE as TSNE


REGULARIZATION = True
TUNED = False
LINEAR = False
PLOT = True

NONLINEAR = True

HYPERPARAM = False
ADAM = False


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

        # if non-linear
        if NONLINEAR:
            # create 30 hidden layers where some are linear and some are non-linear
            self.aa = nn.Linear(k, 100)
            self.ab = nn.Linear(100, 80)
            self.ac = nn.Linear(80, 40)
            self.ad = nn.Linear(40, 20)
            self.ae = nn.Linear(20, 5)
            self.af = nn.ReLU()
            self.ag = nn.Linear(5, 10)
            self.ah = nn.Linear(10, 20)
            self.ai = nn.Linear(20, 40)
            self.aj = nn.Linear(40, 80)
            self.ak = nn.Linear(80, 100)
            self.al = nn.Linear(100, k)
            self.am = nn.ReLU()
            self.an = nn.ReLU()
            self.ao = nn.ReLU()
            self.ap = nn.ReLU()
            self.aq = nn.ReLU()
            self.ar = nn.Linear(k, k)
            self.as_ = nn.Linear(k, k)
            self.at = nn.Linear(k, k)
            self.au = nn.Linear(k, k)
            self.av = nn.Linear(k, k)
            self.aw = nn.Linear(k, k)

        self.h = nn.Linear(k, num_question)

    def get_weight_norm(self):
        """ Return ||W^1||^2 + ||W^2||^2.

        :return: float
        """
        g_w_norm = torch.norm(self.g.weight.clone().detach(), 2) ** 2
        h_w_norm = torch.norm(self.h.weight.clone().detach(), 2) ** 2
        if NONLINEAR:
            aa_w_norm = torch.norm(self.aa.weight.clone().detach(), 2) ** 2
            ab_w_norm = torch.norm(self.ab.weight.clone().detach(), 2) ** 2
            ac_w_norm = torch.norm(self.ac.weight.clone().detach(), 2) ** 2
            ad_w_norm = torch.norm(self.ad.weight.clone().detach(), 2) ** 2
            ae_w_norm = torch.norm(self.ae.weight.clone().detach(), 2) ** 2
            # af_w_norm = torch.norm(self.af.weight.clone().detach(), 2) ** 2
            ag_w_norm = torch.norm(self.ag.weight.clone().detach(), 2) ** 2
            ah_w_norm = torch.norm(self.ah.weight.clone().detach(), 2) ** 2
            ai_w_norm = torch.norm(self.ai.weight.clone().detach(), 2) ** 2
            aj_w_norm = torch.norm(self.aj.weight.clone().detach(), 2) ** 2
            ak_w_norm = torch.norm(self.ak.weight.clone().detach(), 2) ** 2
            al_w_norm = torch.norm(self.al.weight.clone().detach(), 2) ** 2
            # am_w_norm = torch.norm(self.am.weight.clone().detach(), 2) ** 2
            # an_w_norm = torch.norm(self.an.weight.clone().detach(), 2) ** 2
            # ao_w_norm = torch.norm(self.ao.weight.clone().detach(), 2) ** 2
            # ap_w_norm = torch.norm(self.ap.weight.clone().detach(), 2) ** 2
            # aq_w_norm = torch.norm(self.aq.weight.clone().detach(), 2) ** 2
            ar_w_norm = torch.norm(self.ar.weight.clone().detach(), 2) ** 2
            as_w_norm = torch.norm(self.as_.weight.clone().detach(), 2) ** 2
            at_w_norm = torch.norm(self.at.weight.clone().detach(), 2) ** 2
            au_w_norm = torch.norm(self.au.weight.clone().detach(), 2) ** 2
            av_w_norm = torch.norm(self.av.weight.clone().detach(), 2) ** 2
            aw_w_norm = torch.norm(self.aw.weight.clone().detach(), 2) ** 2
            # return g_w_norm + h_w_norm + aa_w_norm + ab_w_norm + ac_w_norm + ad_w_norm + ae_w_norm + af_w_norm + ag_w_norm + ah_w_norm + ai_w_norm + aj_w_norm + ak_w_norm + al_w_norm + am_w_norm + an_w_norm + ao_w_norm + ap_w_norm + aq_w_norm + ar_w_norm + as_w_norm + at_w_norm + au_w_norm + av_w_norm + aw_w_norm
            return g_w_norm + h_w_norm + aa_w_norm + ab_w_norm + ac_w_norm + ad_w_norm + ae_w_norm + ag_w_norm + ah_w_norm + ai_w_norm + aj_w_norm + ak_w_norm + al_w_norm + ar_w_norm + as_w_norm + at_w_norm + au_w_norm + av_w_norm + aw_w_norm
        return g_w_norm + h_w_norm

    def forward(self, inputs):
        """ Return a forward pass given inputs.

        :param inputs: user vector.
        :return: user vector.
        """
        #####################################################################                                                   #
        # Implement the function as described in the docstring.             #
        # Use sigmoid activations for f and g.                              #
        #####################################################################

        # relu vs simgoid
        if NONLINEAR:
            x = torch.sigmoid(self.g(inputs))
            x = torch.sigmoid(self.aa(x))
            x = torch.sigmoid(self.ab(x))
            x = torch.sigmoid(self.ac(x))
            x = torch.sigmoid(self.ad(x))
            x = torch.sigmoid(self.ae(x))
            x = torch.sigmoid(self.af(x))
            x = torch.sigmoid(self.ag(x))
            x = torch.sigmoid(self.ah(x))
            x = torch.sigmoid(self.ai(x))
            x = torch.sigmoid(self.aj(x))
            x = torch.sigmoid(self.ak(x))
            x = torch.sigmoid(self.al(x))
            x = torch.sigmoid(self.am(x))
            x = torch.sigmoid(self.an(x))
            x = torch.sigmoid(self.ao(x))
            x = torch.sigmoid(self.ap(x))
            x = torch.sigmoid(self.aq(x))
            x = torch.sigmoid(self.ar(x))
            x = torch.sigmoid(self.as_(x))
            x = torch.sigmoid(self.at(x))
            x = torch.sigmoid(self.au(x))
            x = torch.sigmoid(self.av(x))
            x = torch.sigmoid(self.aw(x))
            x = torch.sigmoid(self.h(x))
            out = x
        else:

            x = torch.sigmoid(self.g(inputs))
            x = torch.sigmoid(self.h(x))
            out = x

        #####################################################################
        #                       END OF YOUR CODE                            #
        #####################################################################
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
    if REGULARIZATION:
        weight_norm = (lamb / 2) * model.get_weight_norm()
    else:
        weight_norm = 0

    # Tell PyTorch you are training the model.
    model.train()

    # Define optimizers and loss function.
    optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=lamb)

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
            loss = torch.sum((output - target) ** 2) + weight_norm
            loss.backward(retain_graph=True)

            train_loss += loss.item()
            optimizer.step()

        valid_acc = evaluate(model, zero_train_data, valid_data)
        valid_accuracies.append(valid_acc)
        train_cost.append(train_loss)
        print("Epoch: {} \tTraining Cost: {:.6f}\t "
              "Valid Acc: {}".format(epoch, train_loss, valid_acc))
    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################
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

    ks = [1, 5, 10, 20, 50, 100, 200, 300, 400, 500]

    # Set optimization hyperparameters.
    lrs = [1, 0.1, 0.01]
    lambs = [0.001, 0.0001, 0.00001, 0.000001, 0.0000001]
    epochs = [3, 4, 5, 6, 7, 8, 9, 10]

    if ADAM:
        max_acc = 0
        best_k = 1
        best_lr = 0.01
        best_lamb = 0.1
        best_epoch = 15
    else:
        max_acc = 0
        best_k = 100
        best_lr = 0.1
        best_lamb = 0.00001
        best_epoch = 10

    if HYPERPARAM:
        for k in ks:
            for lr in lrs:
                for lamb in lambs:
                    for num_epoch in epochs:
                        print("--- k: {} \tlr: {} \tlamb: {} ---".format(k, lr, lamb))

                        if not PLOT:

                            model = AutoEncoder(train_matrix.shape[1], best_k)
                            acc = train(model, best_lr, best_lamb, train_matrix,
                                        zero_train_matrix, valid_data, best_epoch)[0]
                            if max(acc) > max_acc:
                                max_acc = max(acc)
                                best_k = k
                                best_lr = lr
                                best_lamb = best_lamb
                                best_epoch = num_epoch

                            print("Best k: {} \tBest lr: {} \tBest lamb: {} \t Best epoch: {} Best Acc: {}".format(
                                best_k, best_lr, best_lamb, best_epoch, max_acc))

                            if REGULARIZATION:
                                print("Max Accuracy with Reg: {}".format(max_acc))
                            else:
                                print(
                                    "Max Accuracy without Reg: {}".format(max_acc))
    else:
        model = AutoEncoder(train_matrix.shape[1], best_k)
        valid_accs, train_cost = train(model, best_lr, best_lamb, train_matrix,
                                       zero_train_matrix, valid_data, best_epoch)

    print("FINAL: Best k: {} \tBest lr: {} \tBest lamb: {} \t Best epoch: {} Best Acc: {}".format(
        best_k, best_lr, best_lamb, best_epoch, max_acc
    ))

    if PLOT:

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

    #####################################################################
    #                       END OF YOUR CODE                            #
    #####################################################################


if __name__ == "__main__":
    main()
