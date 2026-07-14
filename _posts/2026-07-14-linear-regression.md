---
title: "线性回归"
date: 2026-07-14
layout: single
permalink: /deep-learning/
sidebar:
  nav: docs
toc: true
toc_sticky: true
categories:
  - 深度学习
tags:
  - 线性回归
  - 梯度下降
  - PyTorch
mathjax: true
---

## 线性回归的基本元素

*线性回归*（linear regression）基于几个简单的假设：首先，假设自变量 $\mathbf{x}$ 和因变量 $y$ 之间的关系是线性的，即 $y$ 可以表示为 $\mathbf{x}$ 中元素的加权和；这里通常允许包含观测值的一些[^1]噪声。其次，我们假设噪声较为正常，例如服从正态分布。

假设有 $n$ 个样本，记

$$
\mathbf{X} =
\begin{bmatrix}
x_1, x_2, \cdots, x_n
\end{bmatrix}^{T},
\qquad
\mathbf{y} =
\begin{bmatrix}
y_1, y_2, \cdots, y_n
\end{bmatrix}^{T}
$$

### 线性模型

- 给定 $n$ 维输入 $\mathbf{X} = \begin{bmatrix}x_1, x_2, \cdots, x_n\end{bmatrix}^{T}$。
- 使用一个 $n$ 维权重和一个标量偏置：$\mathbf{w} = [w_1, w_2, \ldots, w_n]^T,\ b$。
- 输出是输入的加权和：$y = w_1x_1 + w_2x_2 + \cdots + w_nx_n + b = \langle \mathbf{x}, \mathbf{w} \rangle + b$。

### 损失函数

*损失函数*（loss function）能够量化目标的*实际值*与*预测值*之间的差距。

<p align="center">
  <img src="{{ '/images/fit-linreg.png' | relative_url }}" alt="线性回归损失函数示意图">
</p>

假设 $y$ 是真实值，$\hat{y}$ 是估计值，可以计算平方损失：

$$
\ell(y, \hat{y}) = \frac{1}{2}(y - \hat{y})^2。
$$

- 训练损失：

$$
\ell(X, y, w, b)
= \frac{1}{2n} \sum_{i=1}^{n} \left(y_i - (\langle x_i, w \rangle - b)\right)^2
= \frac{1}{2n} \| y - Xw - b \|^2
$$

- 最小化损失：

$$
w^*, b^* = \arg\min_{w,b} \, L(w,b)
$$

### 解析解

将偏置项加入权重：

$$
x \leftarrow [x,1], \quad w \leftarrow \begin{bmatrix} w \\ b \end{bmatrix}
$$

$$
\ell(X, y, w) = \frac{1}{2n}\|y - Xw\|^2
\qquad
\frac{\partial}{\partial w}\ell(X,y,w)
= \frac{1}{n}(Xw - y)^{\top}X
= \frac{1}{n}X^{\top}(Xw - y)
$$

损失是凸函数，因此最优解满足：

$$
\frac{\partial}{\partial w}\ell(X,y,w)=0
$$

$$
\nabla_w \ell
= \frac{1}{n}X^{\top}(Xw-y)=0
\;\Rightarrow\;
X^{\top}Xw=X^{\top}y
\;\Rightarrow\;
w^*=(X^{\top}X)^{-1}X^{\top}y
$$

## 基础优化方法

### 梯度下降

示例：最小化 $f(x) = x^2$。

- 选择一个初始值 $x_0 = 4$，学习率 $\eta = 0.1$。
- 重复迭代参数，$t = 0, 1, 2, \ldots$。

$$
\nabla f(x_t) = f'(x_t) = 2x_t
\qquad
x_{t+1} = x_t - \eta \cdot f'(x_t)
$$

<p align="center">
  <img src="{{ '/images/选择学习率.png' | relative_url }}" width="339" alt="学习率示意图">
</p>

### 小批量随机梯度下降

在整个训练集上计算的代价很高：一个深度神经网络模型可能需要数分钟至数小时。我们可以随机采样 $b$ 个样本 $i_1, i_2, \cdots, i_b$ 来近似损失：

$$
\frac{1}{b}\sum_{i \in I_b} \ell(x_i, y_i, \mathbf{w})
$$

> - 梯度下降通过不断沿着负梯度方向更新参数求解。
> - 小批量随机梯度下降是深度学习默认的求解算法。
> - 两个重要的超参数是**批量大小** $b$ 和**学习率** $\eta$。

## 从零开始实现

### 准备工作

1. `%matplotlib inline`：将 [^2]Matplotlib 绘制的图像直接显示在当前 Notebook 单元格中，而不是弹出新窗口。
2. `import random`：导入 Python 标准库 `random`，用于生成随机数等。
3. `import matplotlib.pyplot as plt`：导入 Matplotlib 的 `pyplot` 模块，并命名为 `plt`，用于绘图。

### 生成数据集

生成一个包含 1000 个样本的数据集，每个样本包含从标准正态分布中采样的 2 个特征。合成数据集是一个矩阵 $\mathbf{X}\in\mathbb{R}^{1000\times2}$。

使用线性模型参数 $\mathbf{w} = \begin{bmatrix}2,-3.4\end{bmatrix}^{\mathsf{T}}$、$b = 4.2$ 和噪声项生成数据集及其标签：

$$
y = \mathbf{X}\mathbf{w} + b + \varepsilon.
$$

$\varepsilon$ 可以视为模型预测和标签之间的潜在观测误差。这里认为标准假设成立，即 $\varepsilon$ 服从均值为 0 的正态分布。为简化问题，将标准差设为 0.01。

```python
def synthetic_data(w, b, num_examples):
    X = torch.normal(0, 1, (num_examples, len(w)))  # 从正态分布中随机采样
    y = torch.matmul(X, w) + b  # 矩阵乘法
    y += torch.normal(0, 0.01, y.shape)
    return X, y.reshape((-1, 1))  # -1 表示 PyTorch 自动判断该维度大小

true_w = torch.tensor([2, -3.4])
true_b = 4.2
features, labels = synthetic_data(true_w, true_b, 1000)
```

通过绘制第二个特征 `features[:, 1]` 和 `labels` 的散点图，可以直观地观察二者之间的线性关系。

```python
plt.figure(figsize=(3.5, 2.5))  # 设置图像大小（单位：英寸）
plt.scatter(features[:, 1].detach().numpy(), labels.detach().numpy(), 1)
```

`detach()` 的作用是将张量从计算图中分离出来。Matplotlib 不需要梯度；若直接使用带梯度的张量，可能导致内存占用上升、阻碍 `.numpy()` 转换或污染梯度链路。

<p align="center">
  <img src="{{ '/images/output_linear-regression-scratch_58de05_51_0.png' | relative_url }}" alt="合成数据集散点图">
</p>

### 读取数据集

训练模型时需要遍历数据集。我们可以每次**抽取一小批量样本**，再使用它们更新模型。由于这个过程是训练机器学习算法的基础，有必要定义一个能够**打乱数据集中的样本**并以小批量方式获取数据的函数。

下面的 `data_iter` 函数接收批量大小、特征矩阵和标签向量作为输入，生成大小为 `batch_size` 的小批量；每个小批量包含一组特征和标签。

```python
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))  # 生成所有样本的索引
    random.shuffle(indices)  # 每个 epoch 都打乱数据顺序
    for i in range(0, num_examples, batch_size):
        batch_indices = torch.tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield features[batch_indices], labels[batch_indices]
```

通常，我们利用 [^3]GPU 并行运算的优势处理合理大小的“小批量”。每个样本可以并行地进行模型计算，每个样本损失函数的梯度也可以并行计算。

运行迭代时，会连续获得不同的小批量，直至遍历完整个数据集。上面实现的迭代器适合教学，但执行效率较低，可能要求将所有数据加载到内存中并执行大量随机内存访问。深度学习框架中实现的**内置迭代器**效率更高，可以处理存储在文件中的数据和数据流提供的数据。

### 初始化模型参数

在使用小批量随机梯度下降优化模型参数之前，需要先有一些参数。下面从均值为 0、标准差为 0.01 的正态分布中采样随机数来初始化权重，并将偏置初始化为 0：

```python
w = torch.normal(0, 0.01, size=(2, 1), requires_grad=True)
b = torch.zeros(1, requires_grad=True)
```

### 定义模型

```python
def linreg(X, w, b):
    """线性回归模型"""
    return torch.matmul(X, w) + b
```

> 上面的 $\mathbf{X}\mathbf{w}$ 是一个向量，而 $b$ 是一个标量。回想 [2.1.3 节](https://zh-v2.d2l.ai/chapter_preliminaries/ndarray.html#subsec-broadcasting)中描述的广播机制：当我们用一个向量加一个标量时，标量会被加到向量的每个分量上。

### 定义损失函数

因为需要计算损失函数的梯度，所以应先定义损失函数。在实现中，需要将真实值 `y` 的形状转换为和预测值 `y_hat` 相同的形状。

```python
def squared_loss(y_hat, y):
    """均方损失"""
    return (y_hat - y.reshape(y_hat.shape)) ** 2 / 2
```

### 定义优化算法

```python
def sgd(params, lr, batch_size):
    """小批量随机梯度下降"""
    with torch.no_grad():
        for param in params:
            param -= lr * param.grad / batch_size
            param.grad.zero_()
```

1. `with torch.no_grad()`：PyTorch 默认会构建用于自动求导的计算图，但参数更新不是计算图的一部分。因此需要关闭梯度跟踪，否则会将更新步骤错误地纳入反向传播。
2. `param.grad.zero_()`：PyTorch 默认**梯度是累加的**。如果不清零，下一次 `backward` 会叠加上一次的梯度，因此需要清空梯度。

### 训练

训练循环如下：

- 初始化参数。
- 重复训练直到完成：
  - 计算梯度 $g \leftarrow \partial_{(w,b)} \frac{1}{|B|} \sum_{i \in B} l\big(x^{(i)}, y^{(i)}, w, b\big)$。
  - 更新参数 $(w, b) \leftarrow (w, b) - \eta g$。

在每个*迭代周期*（epoch）中，使用 `data_iter` 函数遍历整个数据集，并将训练数据集中的所有样本都使用一次。迭代周期个数 `num_epochs` 和学习率 `lr` 都是超参数，分别设为 3 和 0.03。超参数需要通过反复试验进行调整。

```python
lr = 0.03
num_epochs = 3
net = linreg
loss = squared_loss

for epoch in range(num_epochs):
    for X, y in data_iter(batch_size, features, labels):
        l = loss(net(X, w, b), y)  # X 和 y 的小批量损失
        # l 的形状是 (batch_size, 1)，而不是标量；将所有元素相加，
        # 再计算关于 w 和 b 的梯度。
        l.sum().backward()
        sgd([w, b], lr, batch_size)  # 使用参数的梯度更新参数
    with torch.no_grad():
        train_l = loss(net(features, w, b), labels)
        print(f'epoch {epoch + 1}, loss {float(train_l.mean()):f}')
```

[^1]: 指数据集中存在的随机误差或无关信息。这些噪声会影响数据挖掘模型的准确性和可靠性，可能来源于数据录入错误、传感器误差或数据传输中的干扰等。

[^2]: Matplotlib 是 Python 中经典且常用的数据可视化库，可以将数据绘制成各种图形，帮助我们直观观察数据分布、分析实验结果以及展示模型性能。

[^3]: GPU（Graphics Processing Unit，图形处理器）是一种专门用于并行计算的处理器芯片，最初用于加速图形渲染，现在已广泛用于 AI、科学计算等通用计算场景。
