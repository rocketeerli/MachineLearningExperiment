import numpy as np
from generator import Generator
from matplotlib import pyplot as plt

learningRate = 0.01 # 梯度下降的学习率

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(z))

# 将函数向量化
sigmoid_vec = np.vectorize(sigmoid)

def cal_grad(w, x, y, lamb=0):
    '''
    计算梯度
    '''
    L = y - (1.0 - sigmoid_vec(np.dot(w.T, x)))
    grad = - np.dot(x, L) + lamb * w
    return grad

def gradient_descent(x, y, lamb=0):
    '''
    梯度下降法求系数矩阵 W
    '''
    w = np.zeros((x.shape[0]), dtype=np.float)
    grad = cal_grad(w, x, y, lamb)
    while (np.absolute(grad) > 1e-5).all() :
        w -= grad
        grad = cal_grad(w, x, y, lamb)
    return w

def cal_sigV(w, x):
    V = np.zeros((x.shape[1], x.shape[1]), np.float64)
    L = sigmoid_vec(np.dot(w.T, x))
    for i in range(L.shape[0]):
        V[i, i] = (1 - L[i]) * L[i]
    return V

def newton(x, y, lamb=0):
    '''
    牛顿法求系数矩阵 W   x.shape -> (3, 35)
    '''
    w = np.zeros((x.shape[0]), dtype=np.float)
    grad = cal_grad(w, x, y, lamb)
    V = cal_sigV(w, x)
    H = np.dot(np.dot(x, V), x.T) + lamb
    while (np.absolute(grad) > 1e-6).all() :
        w -= np.dot(np.linalg.inv(H), grad)
        grad = cal_grad(w, x, y, lamb)
        V = cal_sigV(w, x)
        H = np.dot(np.dot(x, V), x.T)
    return w

def draw_line(w, col):
    points_x = np.linspace(-1, 7, 300)
    func = np.poly1d([- w[0] / w[1], -w[2] / w[1]])
    points_y = func(points_x)
    plt.plot(points_x, points_y, color=col)

if __name__ == "__main__":
    gen = Generator()
    x, y = gen.data_generator()
    x = np.hstack((x, [[1] for i in range(x.shape[0])]))

    # 梯度下降法 不带正则项
    w = gradient_descent(x.T, y)
    draw_line(w, 'black')

    # 梯度下降法 带正则项的
    w = gradient_descent(x.T, y, 0.003)
    draw_line(w, 'blue')

    # 牛顿法 不带正则项
    w = newton(x.T, y)
    draw_line(w, 'red')

    # 牛顿法 带正则项
    w = newton(x.T, y, 0.003)
    draw_line(w, 'yellow')

    gen.draw(x)
