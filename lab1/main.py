import numpy as np
from matplotlib import pyplot as plt

# 全局参数
M = 6    # 多项式的阶数
N_train = 10   # 训练数据中，生成的数据数目
N_test = 100   # 测试数据中，生成的数据数目
lamb = 0.00003 # lambda 正则项的系数
learningRate = 0.01 # 梯度下降的学习率

def data_generator(num):
    '''
    生成数据，并添加均值为 0，方差为 1 的高斯噪声
    '''
    x = np.linspace(0, 1, num)
    y = np.sin(2 * np.pi * x)
    y = y + np.random.normal(0.0, 0.3, num)
    return x, y

def cal_X_matrix(M, X):
    ''' 
    根据多项式阶数，计算 X 矩阵 （二维） [a b] -> [[1 a a^2] [1 b b^2]]
    '''
    X_mat = np.zeros((X.shape[0], M+1), dtype=np.float)
    for i in range(X.shape[0]) :
        X_mat[i] = np.array([X[i]**j for j in range(M+1)])
    return X_mat

def cal_analytical_solutions(X, y, lamb=0):
    '''
    计算解析解
    lamb=0时，计算无正则项的解析解; lamb=1时，计算有正则项的解析解，正则项为
    '''
    X_I = np.linalg.inv(np.dot(X.T, X) + lamb * np.identity(X.shape[1]))
    X_mat = np.dot(X_I, X.T)
    W = np.dot(X_mat, y)
    return W

def cal_gradient(W, X, y, lamb):
    '''计算梯度'''
    T = np.dot(X, W) - y
    return np.dot(X.T, T) + lamb * W

def gradient_descent(X, y, lamb, learningRate):
    '''
    梯度下降法求系数矩阵 W
    '''
    W = np.zeros((X.shape[1]), dtype=np.float)
    grad = cal_gradient(W, X, y, lamb)
    while (np.absolute(grad) > 1e-9).all() :
        W = W - learningRate * grad
        grad = cal_gradient(W, X, y, lamb)
    return W

def conjugate_gradient(X, y, lamb, deta=1e-5):
    '''
    共轭梯度
    '''
    A = np.dot(X.T, X) + lamb * np.identity(X.shape[1])
    W = np.zeros((X.shape[1]), dtype=np.float)

    r = -cal_gradient(W, X, y, lamb)  # 以负梯度方向作为初始方向,构造共轭方向
    p = r
    while True:  # 迭代 N 次
        a = np.dot(r.T, r) / np.dot(np.dot(p.T, A), p)
        r_pre = r
        W += a * p
        r -= a * np.dot(A, p)
        if (np.linalg.norm(r) ** 2 < deta):
            break
        p = r + np.dot((np.dot(r.T, r) / np.dot(r_pre.T, r_pre)), p)
    return W


def draw_line(W, num, col):
    '''
    根据系数矩阵，画出拟合后的曲线
    '''
    X_test = np.linspace(0, 1, num)
    X_test_mat = cal_X_matrix(W.shape[0]-1, X_test)
    Y_test = np.dot(X_test_mat, W)
    plt.plot(X_test, Y_test, color = col)

if __name__ == "__main__":
    x, y = data_generator(N_train)   # 生成训练数据
    X = cal_X_matrix(M, x)       # 计算生成矩阵

    # 不带正则项的解析解
    W = cal_analytical_solutions(X, y)  # 计算系数矩阵 W
    draw_line(W, N_test, 'b')

    # 带正则项的解析解
    W = cal_analytical_solutions(X, y, lamb=lamb)  # 计算系数矩阵 W
    draw_line(W, N_test, 'r')

    # 梯度下降法
    W = gradient_descent(X, y, lamb, learningRate)
    draw_line(W, N_test, 'g')

    # 共轭梯度法
    W = conjugate_gradient(X, y, lamb)
    draw_line(W, N_test, 'yellow')

    # 画点
    plt.scatter(x, y)
    plt.xlabel('x')
    plt.ylabel('y')
    plt.show()
