import numpy as np
from generator import Generator
from matplotlib import pyplot as plt

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(z))

# 将函数向量化
sigmoid_vec = np.vectorize(sigmoid)

class Logistic:
    def __init__(self, x, y, learningRate=0.01, lamb=0):
        self.x = x                           # 样本特征
        self.y = y                           # 样本类别
        self.learningRate = learningRate     # 梯度下降的学习率
        self.lamb = lamb                     # 正则项的系数

    def cal_grad(self, w):
        '''
        计算梯度
        '''
        L = self.y - (1.0 - sigmoid_vec(np.dot(w.T, self.x)))
        grad = - np.dot(self.x, L) + self.lamb * w
        return grad

    def gradient_descent(self):
        '''
        梯度下降法求系数矩阵 W
        '''
        w = np.zeros((self.x.shape[0]), dtype=np.float)
        grad = self.cal_grad(w)
        while (np.absolute(grad) > 1e-5).all() :
            w -= grad
            grad = self.cal_grad(w)
        return w

    def cal_sigV(self, w):
        V = np.zeros((self.x.shape[1], self.x.shape[1]), np.float64)
        L = sigmoid_vec(np.dot(w.T, self.x))
        for i in range(L.shape[0]):
            V[i, i] = (1 - L[i]) * L[i]
        return V

    def newton(self):
        '''
        牛顿法求系数矩阵 W   x.shape -> (3, 35)
        '''
        w = np.zeros((self.x.shape[0]), dtype=np.float)
        grad = self.cal_grad(w)
        V = self.cal_sigV(w)
        H = np.dot(np.dot(self.x, V), self.x.T) + self.lamb
        while (np.absolute(grad) > 1e-6).all() :
            w -= np.dot(np.linalg.inv(H), grad)
            grad = self.cal_grad(w)
            V = self.cal_sigV(w)
            H = np.dot(np.dot(self.x, V), self.x.T)
        return w

if __name__ == "__main__":
    print("Please run the main file")
