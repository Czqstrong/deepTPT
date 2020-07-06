# Import packages
import matplotlib.pyplot as plt
import numpy as np
from KDEpy import FFTKDE
from scipy import stats
from scipy.integrate import dblquad
from bandwidth import kde2d

# Initial data
num = 10 ** 7
length = 10
x = np.zeros([length, num])
y = np.zeros([length, num])
x[:, 0] = stats.norm(loc=0, scale=1).rvs(length)
y[:, 0] = stats.norm(loc=0, scale=1).rvs(length)
b1 = np.zeros(length)
b2 = np.zeros(length)
s1 = np.sqrt(2)
s2 = np.sqrt(2)

# Number of points in partition
N = 10 ** 7

# Time increments
dt = 0.001
norm = stats.norm(loc=0, scale=np.sqrt(dt))

# Iterate
for i in range(1, N):
    b1 = -2 * x[:, i - 1] * dt
    b2 = -2 * y[:, i - 1] * dt
    # b1 = -10 * x[:, i - 1] * (x[:, i - 1] ** 2 - 1) * dt
    # b2 = -10 * y[:, i - 1] * dt
    x[:, i] = x[:, i - 1] + b1 + s1 * norm.rvs(length)
    y[:, i] = y[:, i - 1] + b2 + s2 * norm.rvs(length)
    print(i)

x_id = x[:, 10 ** 6:num + 1]  # 去掉前面1000个点
y_id = y[:, 10 ** 6:num + 1]
nd = length * (num - 10 ** 6)
x_d = x_id.reshape(1, nd)
y_d = y_id.reshape(1, nd)
plt.scatter(x_d, y_d)
nn = len(x_d)
# kernel density estimation
data = np.vstack((x_d, y_d)).T
cov = np.cov(data.T)
grid_points = 2 ** 7  # Grid points in each dimension
N = 16  # Number of contours

# bd = kde2d(data, nn)
# # Compute the kernel density estimate
# bandwidth = bd.compute()
kde = FFTKDE(bw=0.05, norm=2)
grid, points = kde.fit(data).evaluate(grid_points)

# The grid is of shape (obs, dims), points are of shape (obs, 1)
xx, yy = np.unique(grid[:, 0]), np.unique(grid[:, 1])
x, y = np.meshgrid(xx, yy)
# Z, err = dblquad(lambda y, x: np.exp(-2.5 * (x ** 2 - 1) ** 2 - 5 * y ** 2), -np.inf, np.inf, lambda x: -np.inf,
#                  lambda x: np.inf)
# p = np.exp(-2.5 * (grid[:, 0] ** 2 - 1) ** 2 - 5 * grid[:, 1] ** 2) / Z
Z, err = dblquad(lambda y, x: np.exp(-x ** 2 - y ** 2), -np.inf, np.inf, lambda x: -np.inf,
                 lambda x: np.inf)
p = np.exp(- grid[:, 0] ** 2 - grid[:, 1] ** 2) / Z
error = points - p
z1 = points.reshape(grid_points, grid_points).T
z2 = p.reshape(grid_points, grid_points).T
z3 = error.reshape(grid_points, grid_points).T

# Plot the kernel density estimate
fig = plt.figure()
# ax1 = fig.add_subplot(1, 3, 1, projection='3d')
# ax1.plot_surface(x, y, z1, alpha=0.3, cmap='winter')
# ax1.contourf(x, y, z1, zdir='z', offset=-3, cmap="rainbow")
# ax1.contourf(x, y, z1, zdir='x', offset=-6, cmap="rainbow")
# ax1.contourf(x, y, z1, zdir='y', offset=6, cmap="rainbow")
# ax1.set_xlabel('X')
# ax1.set_xlim(-6, 6)
# ax1.set_ylabel('Y')
# ax1.set_ylim(-6, 6)
# ax1.set_zlabel('Z')
# ax1.set_zlim(0, 2)
ax1 = fig.add_subplot(1, 3, 1)
plt.contourf(x, y, z1, N, cmap="RdBu_r")
plt.colorbar()
# ax2 = fig.add_subplot(1, 3, 2, projection='3d')
# ax2.plot_surface(x, y, z1, alpha=0.3, cmap='winter')
# ax2.contourf(x, y, z1, zdir='z', offset=-3, cmap="rainbow")
# ax2.contourf(x, y, z1, zdir='x', offset=-6, cmap="rainbow")
# ax2.contourf(x, y, z1, zdir='y', offset=6, cmap="rainbow")
# ax2.set_xlabel('X')
# ax2.set_xlim(-6, 6)
# ax2.set_ylabel('Y')
# ax2.set_ylim(-6, 6)
# ax2.set_zlabel('Z')
# ax2.set_zlim(0, 2)
ax2 = fig.add_subplot(1, 3, 2)
plt.contourf(x, y, z2, N, cmap="RdBu_r")
plt.colorbar()
ax3 = fig.add_subplot(1, 3, 3)
plt.contourf(xx, yy, z3, N, cmap="RdBu_r")
plt.colorbar()
plt.show()

