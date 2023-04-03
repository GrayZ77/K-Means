# K-Means算法实现图像分割实验报告

201250008 徐瀚林

[TOC]

## 1. K-Means算法阐述

K-均值是一个迭代算法，假设我们想要将数据聚类成 n 个组，其方法为:

· 首先选择𝐾个随机的点，称为聚类中心（cluster centroids）；
· 对于数据集中的每一个数据，按照距离𝐾个中心点的距离，将其与距离最近的中心点关联起来，与同一个中心点关联的所有点聚成一类。
· 计算每一个组的平均值，将该组所关联的中心点移动到平均值的位置。
· 重复步骤，直至中心点不再变化或达到最大迭代次数（本次使用）。

## 2. 核心代码解释

### 2.1 数据载入

```python
img = plt.imread('1.jpg')
```

读入图片，并以三维数组的形式存储（高，宽， 3）

3是特征数，图片即为RGB属性

由于K-means算法需要的数据应为样本数×特征数，因此进行以下变换

```python
img = img.reshapt(-1, 3)
```

最后，我们新增一列，来标识每个像素点的聚类信息

```python
img = np.column__stack((img, np.ones(row*col)))
```

### 2.2 质心选择

开始时，随机选择[0, col*row-1]范围内的任意k个数，作为索引，从data（数据集）中根据索引，找到初始质心

```python
cluster_center = data[np.random.choice(row*col, k)]
#row = data.shape[0] 行数
#col = data.shape[1] 列数
```

随后计算每个点到这k个质心的欧氏距离

```python
distance = np.sqrt(np.sum((x - y)**2, axis=1))
```

将距离最近的质心的索引放入载入数据时添加的第三列中

```python
data[:, 3] = np.argmin(distance, axis=0)
```

最后计算全部k个聚类的新质心（即该聚类中的所有点的均值）】

```python
for j in range(k):
	cluster.center[j] = np.mean(data[data[:, 3] == j], axis=0)
```

随后开始迭代这一过程

### 2.3 结果可视化

K-means方法的返回值应当返回所有点的聚类结果：

```python
	return img[:, 3]
```

随后，将这一数组重构为图像属性的数组

```python
image_show = image_show.reshape(row, col)
```

最后，利用matplot绘制图像

```python
plt.subplot(122)
plt.imshow(image_show)
plt.show()
```



## 3. 图像可视化及结果展示

原数据如下：

![1](./1.jpg)

迭代100次，k=5的运行结果：

![结果1](./结果1.png)

迭代50次，k=7的运行结果：

![结果2](./结果2.png)

迭代100次，k=10的运行结果：

![结果3](./结果3.png)
