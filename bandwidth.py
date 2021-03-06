# Import packages
import numpy as np
from scipy.optimize import brentq
import logging
from scipy.fftpack import dct
import matplotlib.pyplot as plt
from scipy.integrate import dblquad


class kde2d:
    def __init__(self, data, N):
        # data is a 2d vector
        # Initial data for KDE
        self.n = 2 ** 8
        self.N = N  # number of the initial data
        self.MAX = np.max(data, axis=0)
        self.MIN = np.min(data, axis=0)
        self.Range = self.MAX - self.MIN
        self.MAX_XY = self.MAX + self.Range / 2
        self.MIN_XY = self.MIN - self.Range / 2
        self.scaling = self.MAX_XY - self.MIN_XY
        # 将数据归一化到[0,1]区间内
        # bin the data uniformly using regular grid
        transformed_data = (data - self.MIN_XY) / np.tile(self.scaling, (self.N, 1))
        self.initial_data = self.ndhist(transformed_data, self.n)
        self.a = self.dct2d(self.initial_data)
        self.I = np.arange(0, self.n, dtype=np.float64) ** 2
        self.A2 = self.a ** 2

    # this function computes the histogram
    # of an n-dimensional data set
    # 'data' is nrows by n columns
    # M is the number of bins used in each dimension
    # so that 'binned_data' is a hypercube with
    # size length equal to M
    def ndhist(self, data, M):
        [nrows, _] = np.shape(data)
        binned_data, _, _ = np.histogram2d(data[:, 0].T, data[:, 1].T, bins=(np.linspace(0, 1, M + 1),
                                                                             np.linspace(0, 1, M + 1)),
                                           weights=np.ones_like(data[:, 0]) * (1 / nrows))
        return binned_data

    # discrete cosine transform of initial data
    def dct1d(self, x, weights):
        x = np.vstack((x[::2, :], x[::-2, :]))
        transform1d = (weights * np.fft.fft(x, axis=0)).real
        return transform1d

    def dct2d(self, data):
        [nrows, ncols] = np.shape(data)
        w1 = np.array([2 * np.exp(-1j * i * np.pi / (2 * nrows)) for i in range(1, nrows)])[:, np.newaxis]
        w = np.vstack((np.array([1]), w1))
        weight = np.tile(w, (1, ncols))
        data = self.dct1d(self.dct1d(data, weight).T, weight).T
        return data

    def idct1d(self, x, weights, nrows, ncols):
        y = np.fft.ifft(weights * x, axis=0).real
        out = np.zeros([nrows, ncols])
        idx = int(nrows / 2)
        out[0:nrows:2, :] = y[0:idx, :]
        out[1:nrows:2, :] = y[nrows - 1:idx - 1:-1, :]
        return out

    def idct2d(self, data):
        [nrows, ncols] = np.shape(data)
        w = np.array([np.exp(1j * i * np.pi / (2 * nrows)) for i in range(0, nrows)])[:, np.newaxis]
        weights = np.tile(w, (1, ncols))
        data = self.idct1d(self.idct1d(data, weights, nrows, ncols).T, weights, nrows, ncols)
        return data

    def func(self, s, t):
        sums = np.sum(s)
        if sums <= 4:
            sum_func = self.func([s[0] + 1, s[1]], t) + self.func([s[0], s[1] + 1], t)
            const = (1 + 0.5 ** (sums + 1)) / 3
            time = (-2 * const * self.K(s[0]) * self.K(s[1]) / self.N / sum_func) ** (1. / (2 + sums))
            return self.psi(s, time)
        else:
            return self.psi(s, t)

    def psi(self, s, time):
        pisquared = np.pi ** 2
        tn = 0.5 * np.ones(self.n)
        tn[0] = 1
        w = np.exp(-self.I * pisquared * time) * tn
        wx = w * (self.I ** s[0])
        wy = w * (self.I ** s[1])
        return (-1) ** np.sum(s) * wy.dot(self.A2).dot(wx.T) * np.pi ** (2 * np.sum(s))

    def K(self, s):
        return (-1) ** s * np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2 * np.pi)

    def _bandwidth_fixed_point_2D(self, t):
        sum_func = self.func([0, 2], t) + self.func([2, 0], t) + 2 * self.func([1, 1], t)
        time = (2 * np.pi * self.N * sum_func) ** (-1. / 3)
        return (t - time) / time

    # compute the optimal bandwidth
    def compute(self, fallback_t=None):
        # now compute the optimal bandwidth^2
        try:
            # t is the bandwidth squared (used for estimating moments), calculated using fixed point
            self.t_star = brentq(self._bandwidth_fixed_point_2D, 0, 0.2, xtol=0.001 ** 2)
            # noinspection PyTypeChecker
            if fallback_t and self.t_star > 0.01 and self.t_star > 2 * fallback_t:
                # For 2D distributions with boundaries, fixed point can overestimate significantly
                logging.debug('KernelOptimizer2D Using fallback (t* > 2*t_gallback)')
                self.t_star = fallback_t
        except Exception:
            if fallback_t is not None:
                # Note the fallback result actually appears to be better in some cases,
                # e.g. Gaussian with four cuts
                logging.debug('2D kernel density optimizer using fallback plugin width %s' % (np.sqrt(fallback_t)))
                self.t_star = fallback_t
            else:
                raise
        p_02 = self.func([0, 2], self.t_star)
        p_20 = self.func([2, 0], self.t_star)
        p_11 = self.func([1, 1], self.t_star)
        t_x = ((p_02 ** (3 / 4)) / (4 * np.pi * self.N * (p_20 ** (3 / 4) * (p_11 + np.sqrt(p_20 * p_02))))) ** (1 / 3)
        t_y = ((p_20 ** (3 / 4)) / (4 * np.pi * self.N * (p_02 ** (3 / 4) * (p_11 + np.sqrt(p_20 * p_02))))) ** (1 / 3)
        a1 = np.array([np.exp(-(i ** 2) * (np.pi ** 2) * (t_x / 2)) for i in range(0, self.n)])[:, np.newaxis]
        a2 = np.array([np.exp(-(i ** 2) * np.pi ** 2 * (t_y / 2)) for i in range(0, self.n)])[:, np.newaxis]
        a2 = a2.T
        a_t = a1.dot(a2) * self.a
        density = self.idct2d(a_t) * np.size(self.a) / np.prod(self.scaling)
        idx = (density < 0)
        density[idx] = np.spacing(1)
        x = np.linspace(self.MIN_XY[0], self.MAX_XY[0], self.n)
        y = np.linspace(self.MIN_XY[1], self.MAX_XY[1], self.n)
        [X, Y] = np.meshgrid(x, y)
        # Z, err = dblquad(lambda y, x: np.exp(-2.5 * (x ** 2 - 1) ** 2 - 5 * y ** 2), -np.inf, np.inf, lambda x: -np.inf,
        #                  lambda x: np.inf)
        Za, _ = dblquad(lambda y, x: np.exp(-2.5 * (x ** 2 - 1) ** 2 - 5 * y ** 2), self.MIN_XY[0], self.MAX_XY[0],
                        lambda x: self.MIN_XY[1], lambda x: self.MAX_XY[1])
        Tpoints = np.exp(-2.5 * (X ** 2 - 1) ** 2 - 5 * Y ** 2) / Za
        error = Tpoints - density
        fig = plt.figure()
        ax1 = fig.add_subplot(1, 3, 1)
        plt.contourf(X, Y, density, cmap="RdBu_r")
        plt.colorbar()
        ax2 = fig.add_subplot(1, 3, 2)
        plt.contourf(X, Y, Tpoints, cmap="RdBu_r")
        plt.colorbar()
        ax3 = fig.add_subplot(1, 3, 3)
        plt.contourf(X, Y, error, cmap="RdBu_r")
        plt.colorbar()
        plt.show()
        return np.sqrt([t_x, t_y]) * self.scaling


class kde3d:
    def __init__(self, data, N):
        # Initial data for KDE
        self.n = 2 ** 8
        self.N = N  # number of the initial data
        self.MAX = np.max(data, axis=0)
        self.MIN = np.min(data, axis=0)
        self.Range = self.MAX - self.MIN
        self.MAX_XY = self.MAX + self.Range / 2
        self.MIN_XY = self.MIN - self.Range / 2
        self.scaling = self.MAX_XY - self.MIN_XY
        # 将数据归一化到[0,1]区间内
        # bin the data uniformly using regular grid
        transformed_data = (data - self.MIN_XY) / np.tile(self.scaling, (self.N, 1))
        self.initial_data = self.ndhist(transformed_data, self.n)
        a = self.dct3d(self.initial_data)
        self.I = np.arange(0, self.n, dtype=np.float64) ** 2
        self.A2 = a ** 2

    def K(self, s):
        return (-1) ** s * np.prod(np.arange(1, 2 * s, 2)) / np.sqrt(2 * np.pi)

    def dct3d(self, data):
        data = dct(dct(dct(data).transpose(0, 2, 1)).transpose(1, 2, 0)).transpose(1, 2, 0).transpose(0, 2, 1)
        return data

    def ndhist(self, data, M):
        [nrows, _] = np.shape(data)
        binned_data, _ = np.histogramdd(data, bins=(np.linspace(0, 1, M + 1), np.linspace(0, 1, M + 1),
                                                    np.linspace(0, 1, M + 1)),
                                        weights=np.ones_like(data[:, 0]) * (1 / nrows))
        return binned_data

    def func(self, s, t):
        sums = np.sum(s)
        if sums <= 4:
            sum_func = self.func([s[0] + 1, s[1], s[2]], t) + self.func([s[0], s[1] + 1, s[2]], t) \
                       + self.func([s[0], s[1], s[2] + 1], t)
            const = (1 + 0.5 ** (sums + 1.5)) / 3
            time = ((-2 * const * self.K(s[0]) * self.K(s[1]) * self.K(s[2])) / self.N / sum_func) ** (
                    1. / (2.5 + sums))
            return self.psi(s, time)
        else:
            return self.psi(s, t)

    def psi(self, s, time):
        pisquared = np.pi ** 2
        tn = 0.5 * np.ones(self.n)
        tn[0] = 1
        w = np.exp(-self.I * pisquared * time) * tn
        wx = w * (self.I ** s[0])
        wy = w * (self.I ** s[1])
        wz = w * (self.I ** s[2])
        WY, WX, WZ = np.meshgrid(wy, wx, wz)
        W = WX * WY * WZ
        a = (-1) ** np.sum(s) * np.sum(np.sum(W * self.A2)) * np.pi ** (2 * np.sum(s))
        return a

    def _bandwidth_fixed_point_3D(self, t):
        sum_func = self.func([0, 2, 0], t) + self.func([2, 0, 0], t) + self.func([0, 0, 2], t) \
                   + 2 * self.func([1, 1, 0], t) + 2 * self.func([1, 0, 1], t) + 2 * self.func([0, 1, 1], t)
        const = (8 * np.pi ** (1.5)) / 3
        time = (const * self.N * sum_func) ** (-2. / 7)
        return (t - time) / time

    # compute the optimal bandwidth
    def compute(self, fallback_t=None):
        # now compute the optimal bandwidth^2
        try:
            # t is the bandwidth squared (used for estimating moments), calculated using fixed point
            self.t_star = brentq(self._bandwidth_fixed_point_3D, 0, 0.2, xtol=0.01 ** 2)
            # noinspection PyTypeChecker
            if fallback_t and self.t_star > 0.01 and self.t_star > 2 * fallback_t:
                # For 2D distributions with boundaries, fixed point can overestimate significantly
                logging.debug('KernelOptimizer2D Using fallback (t* > 2*t_gallback)')
                self.t_star = fallback_t
        except Exception:
            if fallback_t is not None:
                # Note the fallback result actually appears to be better in some cases,
                # e.g. Gaussian with four cuts
                logging.debug('2D kernel density optimizer using fallback plugin width %s' % (np.sqrt(fallback_t)))
                self.t_star = fallback_t
            else:
                raise
        return np.sqrt(self.t_star) * self.scaling

