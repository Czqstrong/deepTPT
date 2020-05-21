import time

import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import StepLR, MultiStepLR
import numpy as np
import matplotlib.pyplot as plt
import itertools
from pyDOE import lhs


class dgm(torch.nn.Module):
    def __init__(self, m):
        super(dgm, self).__init__()
        self.linear1 = torch.nn.Linear(2, m)
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
        y = F.relu(self.linear1(x))
        y = F.relu(self.linear2(y))
        y = F.relu(self.linear3(y))
        output = self.linear7(y)
        return output


def inretangle(x):
    xmin = [-1.5, -1]
    xmax = [1.5, 1]
    return np.all(x >= xmin) and np.all(x <= xmax)


def inside(x):
    if inretangle(x):
        if ((x[0] + 1) ** 2 + x[1] ** 2 > 0.1) and ((x[0] - 1) ** 2 + x[1] ** 2 > 0.1):
            return True
        else:
            return False
    else:
        return False


def on_boundary1(x):
    if inretangle(x):
        if (x[0] + 1) ** 2 + x[1] ** 2 <= 0.1:
            return True
        else:
            return False
    else:
        return False


def on_boundary2(x):
    if inretangle(x):
        if (x[0] - 1) ** 2 + x[1] ** 2 <= 0.1:
            return True
        else:
            return False
    else:
        return False


def uniform_points(n):
    dim = 2
    xmin = [-1.5, -1]
    xmax = [1.5, 1]
    n1 = int(np.ceil(n ** (1 / dim)))
    xi = []
    for i in range(dim):
        xi.append(np.linspace(xmin[i], xmax[i], num=n1))
    x = np.array(list(itertools.product(*xi)))
    return x


def sample_points(N_u):
    lb = np.array([-1.5, -1])
    ub = np.array([1.5, 1])
    points = lb + (ub - lb) * lhs(2, N_u)
    return points


def pde_loss(xx, yy):
    dy_x = torch.autograd.grad(sum(yy), xx, retain_graph=True, create_graph=True)[0]
    dy_x, dy_y = dy_x[:, 0:1], dy_x[:, 1]
    dy_xx = torch.autograd.grad(sum(dy_x), xx, retain_graph=True, create_graph=True)[0][:, 0]
    dy_yy = torch.autograd.grad(sum(dy_y), xx, retain_graph=True, create_graph=True)[0][:, 1]

    return (dy_xx + dy_yy - 10 * xx[:, 0] * (xx[:, 0] * xx[:, 0] - 1) * dy_x - 10 * xx[:, 1] * dy_y,
            torch.zeros_like(dy_xx))


if __name__ == "__main__":
    N_u1 = 1000
    N_u2 = 1000
    N_s = 30000
    N_f = 2000
    layers = [2, 20, 20, 20, 20, 1]

    X_f_train = uniform_points(N_f)

    # boundary points
    X_u_sample = sample_points(N_s)
    X_u1 = np.array([x1 for x1 in X_u_sample if on_boundary1(x1)])
    idx1 = np.random.choice(X_u1.shape[0], N_u1, replace=False)
    xx1 = X_u1[idx1, :]
    uu1 = np.ones((xx1.shape[0], 1))
    X_u2 = np.array([x2 for x2 in X_u_sample if on_boundary2(x2)])
    idx2 = np.random.choice(X_u2.shape[0], N_u2, replace=False)
    xx2 = X_u2[idx2, :]
    uu2 = np.zeros((xx2.shape[0], 1))

    X_u_train = np.vstack([xx1, xx2])
    u_train = torch.tensor(np.vstack([uu1, uu2]), dtype=torch.float32)
    X_f_train = np.array([x_f for x_f in X_f_train if inside(x_f)])
    x_boundary = torch.tensor(X_u_train, requires_grad=True, dtype=torch.float32)
    x_inner = torch.tensor(X_f_train, requires_grad=True, dtype=torch.float32)

    # define neural network
    mod = dgm(20)

    # optimizer
    optimizer = torch.optim.Adam(mod.parameters(), lr=0.0001)

    # define loss function
    loss_func = torch.nn.MSELoss()

    for t in range(1, 10000):
        yy_inner = mod(x_inner)
        predict_inner, true_inner = pde_loss(x_inner, yy_inner)
        loss_pde = loss_func(predict_inner, true_inner)
        yy_boundary = mod(x_boundary)
        loss_boundary = loss_func(yy_boundary, u_train)
        loss = loss_pde + loss_boundary
        optimizer.zero_grad()  # 清空上一步的残余更新参数值
        loss.backward(retain_graph=True)  # 误差反向传播, 计算参数更新值
        optimizer.step()  # 将参数更新值施加到 net 的 parameters 上
        print(loss.data.numpy())
