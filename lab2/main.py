import numpy as np
from generator import Generator
from logistic import Logistic
from matplotlib import pyplot as plt

def draw_line(w, col):
    points_x = np.linspace(-1, 7, 300)
    func = np.poly1d([- w[0] / w[1], -w[2] / w[1]])
    points_y = func(points_x)
    plt.plot(points_x, points_y, color=col)

if __name__ == "__main__":
    # 生成数据
    gen = Generator()
    x, y = gen.data_generator()
    x = np.hstack((x, [[1] for i in range(x.shape[0])]))

    logist = Logistic(x.T, y)                    # 不带正则项的逻辑回归
    logist_regu = Logistic(x.T, y, lamb=0.003)   # 加入正则项的逻辑回归

    # 梯度下降法 不带正则项
    w = logist.gradient_descent()
    draw_line(w, 'black')

    # 梯度下降法 带正则项的
    w = logist_regu.gradient_descent()
    draw_line(w, 'blue')

    # 牛顿法 不带正则项
    w = logist.newton()
    draw_line(w, 'red')

    # 牛顿法 带正则项
    w = logist_regu.newton()
    draw_line(w, 'yellow')

    gen.draw(x)
