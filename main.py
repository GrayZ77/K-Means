import numpy as np
import matplotlib.pyplot as plt

img = plt.imread('1.jpg')
row = img.shape[0]
col = img.shape[1]
plt.subplot(121)
plt.imshow(img)


def kmeans(data, iter, k):
    data = data.reshape(-1, 3)
    data = np.column_stack((data, np.ones(row * col)))
    # 1.随机产生初始簇心
    cluster_center = data[np.random.choice(row * col, k)]
    # 2.分类
    distance = [[] for i in range(k)]
    for i in range(iter):
        print("迭代次数：", i+1)
        # 2.1距离计算
        for j in range(k):
            distance[j] = np.sqrt(np.sum((data - cluster_center[j]) ** 2, axis=1))
        # 2.2归类
        data[:, 3] = np.argmin(distance, axis=0)
        # 3.计算新簇心
        for j in range(k):
            cluster_center[j] = np.mean(data[data[:, 3] == j], axis=0)
    return data[:, 3]


if __name__ == "__main__":
    image_show = kmeans(img, 100, 10)
    image_show = image_show.reshape(row, col)
    plt.subplot(122)
    plt.imshow(image_show)
    plt.show()