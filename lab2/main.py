import numpy as np
from generator import Generator
from matplotlib import pyplot as plt

learningRate = 0.01 # 梯度下降的学习率

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(z))

# 将函数向量化
sigmoid_vec = np.vectorize(sigmoid)

def gradient_descent(x, y, lamb=0):
    '''
    梯度下降法求系数矩阵 W
    '''
    w = np.zeros((x.shape[0]), dtype=np.float)
    L = y - (1.0 - sigmoid_vec(np.dot(w.T, x)))
    grad = - np.dot(x, L)
    while (np.absolute(grad) > 1e-5).all() :
        w -= grad
        L = y - (1.0 - sigmoid_vec(np.dot(w.T, x)))
        grad = -np.dot(x, L) + lamb * w
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
    w = gradient_descent(x.T, y, 0.0003)
    draw_line(w, 'blue')

    gen.draw(x)
