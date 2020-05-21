import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
from math import *
import matplotlib.pyplot as plt
import matplotlib.cm as cm

torch.set_default_tensor_type('torch.FloatTensor')


class DeepRitzNet(torch.nn.Module):
    def __init__(self, m):
        self.m = m
        super(DeepRitzNet, self).__init__()
        self.linear1 = torch.nn.Linear(m, m)
        self.linear2 = torch.nn.Linear(m, m)
        self.linear3 = torch.nn.Linear(m, m)
        self.linear4 = torch.nn.Linear(m, m)
        self.linear5 = torch.nn.Linear(m, m)
        self.linear6 = torch.nn.Linear(m, m)

        self.linear7 = torch.nn.Linear(m, 1)

        torch.nn.init.constant_(self.linear1.bias, 0.)
        torch.nn.init.constant_(self.linear2.bias, 0.)
        torch.nn.init.constant_(self.linear3.bias, 0.)
        torch.nn.init.constant_(self.linear4.bias, 0.)
        torch.nn.init.constant_(self.linear5.bias, 0.)
        torch.nn.init.constant_(self.linear6.bias, 0.)
        torch.nn.init.constant_(self.linear7.bias, 0.)

        torch.nn.init.normal_(self.linear1.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.linear2.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.linear3.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.linear4.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.linear5.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.linear6.weight, mean=0, std=0.1)
        torch.nn.init.normal_(self.linear7.weight, mean=0, std=0.1)

    def forward(self, x):
        y = x
        y = y + F.relu(self.linear2(F.relu(self.linear1(y))))
        output = F.relu(self.linear7(y))
        return output


def cal_loss(mod):
    points = np.arange(0, 1.1, 0.1)
    xs, ys = np.meshgrid(points, points)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    xl, yl = xs.size()
    z = np.zeros((xl, yl))
    mmm = 0
    t = 0
    for i in range(xl):
        for j in range(yl):
            re = np.zeros(mod.m)
            re[0] = xs[i, j]
            re[1] = ys[i, j]
            re = torch.tensor(re)
            z[i, j] = mod(re.float()).item() - U_groundtruth(re)
            mmm += abs(z[i, j])
            t += 1
    return mmm / t


def relative_err(mod):
    points = np.arange(0, 1, 0.1)
    xs, ys = np.meshgrid(points, points)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    xl, yl = xs.size()
    z = np.zeros((xl, yl))
    w = np.zeros((xl, yl))
    t = 0
    for i in range(xl):
        for j in range(yl):
            re = np.zeros(mod.m)
            re[0] = xs[i, j]
            re[1] = ys[i, j]
            re = torch.tensor(re)
            z[i, j] = mod(re.float()).item() - U_groundtruth(re)
            w[i, j] = U_groundtruth(re)
            t += 1
    z = z ** 2
    w = w ** 2
    return np.sum(z) / np.sum(w)


def U_groundtruth(t):
    return (t[0] * t[1]).item()


def validate(mod):
    draw_graph(mod)
    print(cal_loss(mod))


def draw_graph(mod):
    points = np.arange(0, 1, 0.01)
    xs, ys = np.meshgrid(points, points)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    xl, yl = xs.size()
    z = np.zeros((xl, yl))
    for i in range(xl):
        for j in range(yl):
            re = np.zeros(mod.m)
            re[0] = xs[i, j]
            re[1] = ys[i, j]
            re = torch.tensor(re)
            z[i, j] = mod(re.float()).item() - U_groundtruth(re)

    plt.imshow(z, cmap=cm.hot)
    plt.colorbar()

    my_x_ticks = np.arange(0, 1, 0.2)
    my_y_ticks = np.arange(0, 1, 0.2)
    ax = plt.gca()
    ax.set_xticks(np.linspace(0, 99, 5))
    ax.set_xticklabels(('0', '0.25', '0.5', '0.75', '1'))
    ax.set_yticks(np.linspace(0, 99, 5))
    ax.set_yticklabels(('1', '0.75', '0.5', '0.25', '0'))

    plt.show()


m = 10
learning_rate = 0.01
iterations = 400
print_every_iter = 100
beta = 500  # coefficient for the regularization term in the loss expression
n2 = 100  # number of points on the border of Omega
gamma = 10


def train(mod, opt, initial_lr=learning_rate, n2=n2, milestones=[400], gamma=0.5, alpha=1, alpha_increase=1.01,
          iterations=iterations, beta=1, beta_increase=1.01, print_every_iter=print_every_iter):
    if opt == 'Adam':
        optimizer = torch.optim.Adam(mod.parameters(), lr=initial_lr)
    elif opt == 'SGD':
        optimizer = torch.optim.SGD(mod.parameters(), lr=initial_lr)
    scheduler = MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    mm = 1
    points = np.arange(0, 1, 0.1)
    xs, ys = np.meshgrid(points, points)
    xs = torch.tensor(xs)
    ys = torch.tensor(ys)
    xl, yl = xs.size()

    start = time.time()
    for k in range(iterations):
        n1 = 0
        loss = torch.zeros(1)
        for i in range(xl):
            for j in range(yl):
                x_input = np.zeros(m)
                x_input[0] = xs[i, j]
                x_input[1] = ys[i, j]

                n1 += 1
                x_input = torch.tensor(x_input).float()
                y = mod(x_input)

                x1 = torch.zeros(m)
                x2 = torch.zeros(m)
                x1[0] = 0.0001
                x2[1] = 0.0001
                x_input_1 = x_input.float() + x1
                x_input_2 = x_input.float() + x2
                x_input_3 = x_input.float() - x1
                x_input_4 = x_input.float() - x2
                x_input_grad_1 = (mod(x_input_1) - y) / 0.0001
                x_input_grad_2 = (mod(x_input_2) - y) / 0.0001
                x_input_2_grad_x = (mod(x_input_1) + mod(x_input_3) - 2 * y) / 0.0001 ** 2
                x_input_2_grad_y = (mod(x_input_2) + mod(x_input_4) - 2 * y) / 0.0001 ** 2

                loss = loss + 0.5 * ((x_input_grad_1) ** 2 + (x_input_grad_2) ** 2) \
                       + alpha * (x_input_2_grad_x + x_input_2_grad_y) ** 2
        loss = loss / n1

        regularization = torch.zeros(1)
        for t in range(n2):
            x_input = np.zeros(m)
            x_input[0] = np.random.rand()
            x_input[1] = np.random.rand()
            x_rand = np.random.rand()
            if x_rand < 0.25:
                x_input[0] = 0
            elif x_rand > 0.25 and x_rand < 0.5:
                x_input[0] = 1
            elif x_rand > 0.5 and x_rand < 0.75:
                x_input[1] = 0
            elif x_rand > 0.75:
                x_input[1] = 1
            x_input = torch.tensor(x_input).float()
            y = mod(x_input)
            regularization = regularization + (y - x_input[0] * x_input[1]) ** 2
        regularization = regularization * (beta / n2)

        if alpha < 500:
            alpha = alpha * alpha_increase
        if beta < 500:
            beta = beta * beta_increase

        if ((k + 1) % print_every_iter == 0):
            print(k, " epoch, loss: ", loss.data[0].numpy())
            print(k, " epoch, regularization loss: ", regularization.data[0].numpy())
            print(k, " loss to real solution: ", cal_loss(mod))
        if cal_loss(mod) < 0.0001:
            break

        loss = loss + regularization

        optimizer.zero_grad()
        loss.backward(retain_graph=True)

        scheduler.step()
        optimizer.step()
    stop = time.time()
    print(stop - start)


model = DeepRitzNet(m)
train(model, opt='Adam', iterations=5000, milestones=[10000, 20000, 30000, 40000], gamma=0.5, beta=500,
      beta_increase=1.01, alpha=0, alpha_increase=1, initial_lr=0.00001, print_every_iter=5)
validate(model)
relative_err(model)
