import tensorflow as tf
import numpy as np
import itertools
import time
import matplotlib.pyplot as plt
from pyDOE import lhs

np.random.seed(1234)
tf.set_random_seed(1234)


class PhysicsInformedNN:
    # Initialize the class
    def __init__(self, X_u, u, X_f, layers, lb, ub):

        self.lb = lb
        self.ub = ub

        self.x_u1 = X_u[:, 0:1]
        self.x_u2 = X_u[:, 1:2]

        self.x_f1 = X_f[:, 0:1]
        self.x_f2 = X_f[:, 1:2]

        self.u = u

        self.layers = layers

        # Initialize NNs
        self.weights, self.biases = self.initialize_NN(layers)

        # tf placeholders and graph
        self.sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True,
                                                     log_device_placement=True))

        self.x_u1_tf = tf.placeholder(tf.float32, shape=[None, self.x_u1.shape[1]])
        self.x_u2_tf = tf.placeholder(tf.float32, shape=[None, self.x_u2.shape[1]])
        self.u_tf = tf.placeholder(tf.float32, shape=[None, self.u.shape[1]])

        self.x_f1_tf = tf.placeholder(tf.float32, shape=[None, self.x_f1.shape[1]])
        self.x_f2_tf = tf.placeholder(tf.float32, shape=[None, self.x_f2.shape[1]])

        self.u_pred = self.net_u(self.x_u1_tf, self.x_u2_tf)
        self.f_pred = self.net_f(self.x_f1_tf, self.x_f2_tf)

        self.loss = tf.reduce_mean(tf.square(self.u_tf - self.u_pred)) + \
                    100 * tf.reduce_mean(tf.square(self.f_pred))

        self.optimizer = tf.contrib.opt.ScipyOptimizerInterface(self.loss,
                                                                method='L-BFGS-B',
                                                                options={'maxiter': 50000,
                                                                         'maxfun': 50000,
                                                                         'maxcor': 50,
                                                                         'maxls': 50,
                                                                         'ftol': 1.0 * np.finfo(float).eps})

        self.optimizer_Adam = tf.train.AdamOptimizer()
        self.train_op_Adam = self.optimizer_Adam.minimize(self.loss)
        init = tf.global_variables_initializer()
        self.sess.run(init)

    def initialize_NN(self, layers):
        weights = []
        biases = []
        num_layers = len(layers)
        for l in range(0, num_layers - 1):
            W = self.xavier_init(size=[layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32), dtype=tf.float32)
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim = size[0]
        out_dim = size[1]
        xavier_stddev = np.sqrt(2 / (in_dim + out_dim))
        return tf.Variable(tf.truncated_normal([in_dim, out_dim], stddev=xavier_stddev), dtype=tf.float32)

    def neural_net(self, X, weights, biases):
        num_layers = len(weights) + 1

        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0
        for l in range(0, num_layers - 2):
            W = weights[l]
            b = biases[l]
            H = tf.tanh(tf.add(tf.matmul(H, W), b))
        W = weights[-1]
        b = biases[-1]
        Y = tf.add(tf.matmul(H, W), b)
        return Y

    def net_u(self, x1, x2):
        u = self.neural_net(tf.concat([x1, x2], 1), self.weights, self.biases)
        return u

    def net_f(self, x1, x2):
        u = self.net_u(x1, x2)
        u_x = tf.gradients(u, x1)[0]
        u_y = tf.gradients(u, x2)[0]
        u_xx = tf.gradients(u_x, x1)[0]
        u_yy = tf.gradients(u_y, x2)[0]
        f = u_xx + u_yy - 10 * x1 * (x1 ** 2 - 1) * u_x - 10 * x2 * u_y
        return f

    def callback(self, loss):
        print('Loss:', loss)

    def train(self, nIter):

        tf_dict = {self.x_u1_tf: self.x_u1, self.x_u2_tf: self.x_u2, self.u_tf: self.u,
                   self.x_f1_tf: self.x_f1, self.x_f2_tf: self.x_f2}
        start_time = time.time()
        for it in range(nIter):
            self.sess.run(self.train_op_Adam, tf_dict)

            # Print
            if it % 10 == 0:
                elapsed = time.time() - start_time
                loss_value = self.sess.run(self.loss, tf_dict)
                print('It: %d, Loss: %.3e, Time: %.2f' %
                      (it, loss_value, elapsed))
                start_time = time.time()

        self.optimizer.minimize(self.sess,
                                feed_dict=tf_dict,
                                fetches=[self.loss],
                                loss_callback=self.callback)

    def predict(self, X_star):

        u_star = self.sess.run(self.u_pred, {self.x_u1_tf: X_star[:, 0:1], self.x_u2_tf: X_star[:, 1:2]})
        f_star = self.sess.run(self.f_pred, {self.x_f1_tf: X_star[:, 0:1], self.x_f2_tf: X_star[:, 1:2]})

        return u_star, f_star


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


if __name__ == "__main__":
    noise = 0.0

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
    u_train = np.vstack([uu1, uu2])
    X_f_train = np.array([x_f for x_f in X_f_train if inside(x_f)])

    lb = np.array([-1.5, -1])
    ub = np.array([1.5, 1])

    model = PhysicsInformedNN(X_u_train, u_train, X_f_train, layers, lb, ub)

    start_time = time.time()
    model.train(10)
    elapsed = time.time() - start_time
    print('Training time: %.4f' % elapsed)
    x1 = np.linspace(-1.5, 1.5, 50)
    x2 = np.linspace(-1, 1, 50)
    X1, X2 = np.meshgrid(x1, x2)
    X_star = np.hstack((X1.flatten()[:, None], X2.flatten()[:, None]))
    u_pred, f_pred = model.predict(X_star)
    u_pred = u_pred.reshape(50, 50)
    fig = plt.figure()  # 定义新的三维坐标轴
    ax1 = fig.add_subplot(1, 2, 1, projection='3d')
    ax1.plot_surface(X1, X2, u_pred, cmap='rainbow')
    ax2 = plt.subplot(1, 2, 2)
    plt.contourf(X1, X2, u_pred)
    plt.colorbar()
    plt.show()
