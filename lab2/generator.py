import numpy as np
from matplotlib import pyplot as plt

'''
生成数据
'''
class Generator:
    MEAN_ONE = [1, 1]   # 第一类数据点的均值
    MEAN_TWO = [5, 5]   # 第二类数据点的均值
    COV = [[1, 0.05], [0.05, 1]]  # 两类样本的协方差矩阵（相同的）

    def __init__(self, NUM_ONE=15, NUM_TWO=20):
        self.NUM_ONE = NUM_ONE    # 第一类样本数
        self.NUM_TWO = NUM_TWO    # 第二类样本数

    # 生成两类服从高斯分布的样本
    def data_generator(self):
        x_one = np.random.multivariate_normal(self.MEAN_ONE, self.COV, self.NUM_ONE)
        x_two = np.random.multivariate_normal(self.MEAN_TWO, self.COV, self.NUM_TWO)
        x = np.vstack((x_one, x_two))
        y = np.array([0 for i in range(self.NUM_ONE)] + [1 for i in range(self.NUM_TWO)])
        return x, y

    # 画出两类样本点
    def draw(self, X):
        plt.scatter(X[:self.NUM_ONE,0], X[:self.NUM_ONE,1], c='g')
        plt.scatter(X[self.NUM_ONE:,0], X[self.NUM_ONE:,1], c='b')
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

if __name__ == "__main__":
    gen = Generator()
    x, y = gen.data_generator()
    gen.draw(x)
