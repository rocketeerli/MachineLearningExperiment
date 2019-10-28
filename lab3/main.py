import numpy as np
from generator import Generator

def distance(x, y):
    return np.dot(np.array(x).T, y)

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
        means_pre = means
        # 更新每个类别的均值
        for i in range(k):
            if C[i] == [] :      # 检测类别不为空
                return 0
            means[i] = [np.mean(np.array(C[i])[:,0]), np.mean(np.array(C[i])[:,1])]
        # 当均指不变或超过最大迭代次数时，跳出循环
        if ((means_pre == means).all() or N >= 1000):
            break
        N += 1
    return np.array(C)

if __name__ == "__main__":
    # # 生成数据
    k = 3
    gen = Generator(k)
    x, y = gen.data_generator()

    # gen.draw()

    print(k_means(x, k))