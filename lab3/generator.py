import numpy as np
from matplotlib import pyplot as plt

MEAN = np.array([[0.5, 0.5], [5.5, 5.5], [4, 1], [10, 10], [10, 5], [1, 10], [6, 10], [1, 5], [10, 1]])
COL = ['lime', 'darkviolet', 'forestgreen', 'orange', 'yellow', 'gray', 'blue', 'r', 'black']

# 随机生成协方差矩阵
def rand_cov():
    cov = np.random.rand(2, 2)
    cov += np.eye(*cov.shape)
    cov[1][0] = cov[0][1]
    return cov

# 高斯分布产生 k 个高斯分布的数据（不同均值和方差）
def data_generator(k):
    x = np.random.multivariate_normal(MEAN[0], rand_cov(), 20)
    y = [0 for i in range(20)]
    for i in range(1, k):
        x_i = np.random.multivariate_normal(MEAN[i], rand_cov(), 20)
        x = np.vstack((x, x_i))
        y = y + [i for j in range(20)]
    y = np.array(y)
    return x, y

# 画出 k 类样本点
def draw(x, k):
    for i in range(k):
        plt.scatter(x[20*i:20*(i+1),0], x[20*i:20*(i+1),1], c=COL[i])
    plt.xlabel('x1')
    plt.ylabel('x2')
    plt.show()

if __name__ == "__main__":
    k = 9
    x, y = data_generator(k)
    draw(x, k)
