import numpy as np
from matplotlib import pyplot as plt

'''
生成数据
'''
class Generator:
    MEAN = np.array([[-4, -4], [6, 6], [4, -6], [14, 16], [15, 7], [-3, 10], [15, -8], [-9, 3], [12, -2]])  # 各个类的均值
    COL = ['lime', 'darkviolet', 'forestgreen', 'orange', 'yellow', 'gray', 'blue', 'red', 'black']          # 各类的颜色
 
    def __init__(self, k=3, NUM=20):
        self.k = min(k, 9)                 # 生成训练数据的类别数
        self.NUM = NUM             # 每个类别的数目  

    # 高斯分布产生 k 个高斯分布的数据（不同均值和方差）
    def data_generator(self):
        self.x = np.random.multivariate_normal(self.MEAN[0], self.__rand_cov(), self.NUM)
        self.y = [0 for i in range(self.NUM)]
        for i in range(1, self.k):
            x_i = np.random.multivariate_normal(self.MEAN[i], self.__rand_cov(), self.NUM)
            self.x = np.vstack((self.x, x_i))
            self.y = self.y + [i for j in range(self.NUM)]
        self.y = np.array(self.y)
        return self.x, self.y

    # 画出 k 类样本点
    def draw(self):
        for i in range(self.k):
            plt.scatter(self.x[self.NUM*i:self.NUM*(i+1),0], self.x[self.NUM*i:self.NUM*(i+1),1], c=self.COL[i])
        plt.xlabel('x1')
        plt.ylabel('x2')
        plt.show()

    # 随机生成协方差矩阵
    def __rand_cov(self):
        cov = np.random.rand(2, 2)
        cov += np.eye(*cov.shape)
        cov[1][0] = cov[0][1]
        return cov

if __name__ == "__main__":
    k = 9
    gen = Generator(k)
    x, y = gen.data_generator()
    gen.draw()
