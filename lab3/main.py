import numpy as np
from generator import Generator
from matplotlib import pyplot as plt

COL = ['lime', 'darkviolet', 'forestgreen', 'orange', 'yellow', 'gray', 'blue', 'red', 'black']          # 各类的颜色

def distance(x, y):
    return np.linalg.norm(x-y)     # 求二范式

def k_means(X, k):
    # 随机选取 k 个不同样本作为初始均指向量
    i = np.random.choice(X.shape[0], k)
    means = X[i]
    N = 0
    while True:
        C = [[] for i in range(k)]
        for x in X:
            x = list(x)
            dis = np.array([distance(x, means[j]) for j in range(k)])
            index = np.argmin(dis)
            C[index].append(x)
        means_pre = means.copy()
        # 更新每个类别的均值
        for i in range(k):
            if len(C[i]) == 0:      # 检测类别为空时，赋予之前的类均值
                means[i] = X[np.random.randint(0, X.shape[0], 1)]
            else:
                means[i] = [np.mean(np.array(C[i])[:,0]), np.mean(np.array(C[i])[:,1])]
        # 当均指不变或超过最大迭代次数时，跳出循环
        if ((means_pre == means).all() or N > 100):
            print(N)
            break
        N += 1
    return np.array(C)

def draw(C):
    for i in range(C.shape[0]):
        if len(C[i]) == 0:     # 判断列表是否为空
            continue
        plt.scatter(np.array(C[i])[:,0], np.array(C[i])[:,1], c=COL[i])
    plt.show()

if __name__ == "__main__":
    # 生成数据
    k = 3
    gen = Generator(k, NUM=100) 
    x, y = gen.data_generator()

    # gen.draw()

    C = k_means(x, k)
    draw(C)
