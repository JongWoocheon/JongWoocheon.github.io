---
title: "卷积层和汇聚层"
date: 2026-07-18
layout: single
permalink: /deep-learning/convolutional-and-pooling-layers/
sidebar:
  nav: docs
toc: true
toc_sticky: true
categories:
  - 深度学习
mathjax: true
---

假设有一张有 12M pixels 的 RGB 图片，那么输入维度会变成 36M。如果我们使用大小为 100 的单隐藏层 MLP，即 $W \in \mathbb{R}^{36,000,000 \times 100}$，模型参数有 3.6B，内存占用 14.4GB。想要训练这个模型将不可实现，因为需要有大量的 GPU、分布式优化训练的经验和超乎常人的耐心。

于是我们引入*卷积神经网络*（convolutional neural networks，CNN）。

## 从全连接层到卷积

设一维输入 $x=(x_1,x_2,\cdots,x_n)$，全连接层为 $y=Wx+b$，其中 $W\in R^{m\times n}$。则有 $y_i=\sum_j W_{ij}x_j+b_i$ 。

### 平移不变性

不管检测对象出现在图像中的哪个位置，神经网络的前面几层应该对相同的图像区域具有相似的反应。

如果输入整体移动一个位置：$x_j \rightarrow x_{j-1}$，那么输出应该也移动一个位置：$y_i \rightarrow y_{i-1}$。

$$
W=
\begin{bmatrix} W_{11}&W_{12}&W_{13}&W_{14}&W_{15}\\ W_{21}&W_{22}&W_{23}&W_{24}&W_{25}\\ W_{31}&W_{32}&W_{33}&W_{34}&W_{35}\\ W_{41}&W_{42}&W_{43}&W_{44}&W_{45}\\ W_{51}&W_{52}&W_{53}&W_{54}&W_{55} \end{bmatrix}
\xrightarrow{W_{ij} \ = \ w_{j-i}}
\begin{bmatrix} w_0&w_1&w_2&w_3&w_4\\ w_{-1}&w_0&w_1&w_2&w_3\\ w_{-2}&w_{-1}&w_0&w_1&w_2\\ w_{-3}&w_{-2}&w_{-1}&w_0&w_1\\ w_{-4}&w_{-3}&w_{-2}&w_{-1}&w_0 \end{bmatrix}
$$

平移共享后的公式：$y_i=\sum_k w_kx_{i+k}+b$ 。

### 局部性

神经网络的前面几层应该只探索输入图像中的局部区域，而不过度在意图像中相隔较远区域的关系。

当前位置只依赖附件区域：$\lvert k\rvert \le n$ 。

$$
W=
\begin{bmatrix}
w_0 & w_1 & w_2 & w_3 & w_4\\
w_{-1} & w_0 & w_1 & w_2 & w_3\\
w_{-2} & w_{-1} & w_0 & w_1 & w_2\\
w_{-3} & w_{-2} & w_{-1} & w_0 & w_1\\
w_{-4} & w_{-3} & w_{-2} & w_{-1} & w_0
\end{bmatrix}
\xrightarrow{w_{|k| \ > \ 1} \ = \ 0}
\begin{bmatrix} w_0 & w_1 & 0 & 0 & 0\\ w_{-1} & w_0 & w_1 & 0 & 0\\ 0 & w_{-1} & w_0 & w_1 & 0\\ 0 & 0 & w_{-1} & w_0 & w_1\\ 0 & 0 & 0 & w_{-1} & w_0 \end{bmatrix}
$$

可得到 $y_i=\sum_{k=-r}^{r}w_kx_{i+k}+b$ 。

## 图像卷积

### 互相关运算

其实卷积层是个错误的叫法，它所表达的运算是*互相关运算*（cross-correlation）。

<p align="center">
  <img src="{{ '/images/convolutional-and-pooling-layers_01.png' | relative_url }}"  width="350">
</p>

## 填充和步幅

给定 224×224 的输入图像，在使用 5×5 卷积核的情况下，需要 220÷4 = 55 层才可以将输出降低到 4×4 。

于是我们引入*填充*（padding）和*步幅*（stride）。由于本小节内容比较简单，该笔记仅附上两张照片助于理解。

<div class="image-row">
  <img src="/images/convolutional-and-pooling-layers_02.png">
  <img src="/images/convolutional-and-pooling-layers_03.png">
</div>

## 多输入多输出通道

### 多输入通道

当输入包含多个通道时，需要构造一个与输入数据具有相同输入通道数的卷积核，以便与输入数据进行互相关运算。

在下图中，我们演示了一个具有两个输入通道的二维互相关运算的示例。阴影部分是第一个输出元素以及用于计算这个输出的输入和核张量元素：$(1\times1+2\times2+4\times3+5\times4)+(0\times0+1\times1+3\times2+4\times3)=56$ 。

<p align="center">
  <img src="{{ '/images/convolutional-and-pooling-layers_04.png' | relative_url }}"  width="350">
</p>

### 1×1 卷积层

在不改变特征图空间尺寸的情况下，对通道维度进行线性变换，实现通道融合、降维/升维和增加非线性表达能力。

<p align="center">
  <img src="{{ '/images/convolutional-and-pooling-layers_05.png' | relative_url }}"  width="350">
</p>

## 汇聚层

卷积层对位置敏感。它在检测垂直边缘时，一个像素的移位可能会导致全零的输出，所以我们需要一定程度的平移不变性。

与卷积层类似，汇聚层运算符由一个固定形状的窗口组成，该窗口根据其步幅大小在输入的所有区域上滑动，为固定形状窗口（有时称为*汇聚窗口* ）遍历的每个位置计算一个输出。 然而，不同于卷积层中的输入与卷积核之间的互相关计算，汇聚层不包含参数。 相反，池运算是确定性的，我们通常计算汇聚窗口中所有元素的最大值或平均值。这些操作分别称为*最大汇聚层*（maximum pooling）和*平均汇聚层*（average pooling）。

<p align="center">
  <img src="{{ '/images/convolutional-and-pooling-layers_06.png' | relative_url }}"  width="350">
</p>

<aside class="callout callout--success" role="note">
  <p class="callout__title">✅汇聚层的优势</p>
  <ul>
    <li>降低空间维度</li>
    <li>提升平移不变性，对小范围的位置变化更加稳定</li>
    <li>扩大有效感受野</li>
  </ul>
</aside>

## 代码实现

接下来，我们在`corr2d`函数中实现如上过程，该函数接受输入张量`X`和卷积核张量`K`，并返回输出张量`Y`。

```
def corr2d(X, K):
    """计算二维互相关运算"""
    h, w = K.shape
    Y = torch.zeros((X.shape[0] - h + 1, X.shape[1] - w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            Y[i, j] = (X[i:i + h, j:j + w] * K).sum()
    return Y
```

基于上面定义的`corr2d`函数实现二维卷积层。在`__init__`构造函数中，将`weight`和`bias`声明为两个模型参数。前向传播函数调用`corr2d`函数并添加偏置。

```
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(kernel_size))
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return corr2d(x, self.weight) + self.bias
```

我们定义了一个计算卷积层的函数。此函数初始化卷积层权重，并对输入和输出提高和缩减相应的维数 。

```
def comp_conv2d(conv2d, X):
    X = X.reshape((1, 1) + X.shape)  # 这里的 (1, 1) 表示批量大小和通道数都是 1
    Y = conv2d(X)
    return Y.reshape(Y.shape[2:])  # 省略前两个维度：批量大小和通道

X = torch.rand(size=(8, 8))
# 输入 1 个通道；使用 1 个卷积核；左右各填充 1 个像素；高度：每次移动 3 个像素；宽度：每次移动 4 个像素
conv2d = nn.Conv2d(1, 1, kernel_size=(3, 5), padding=(0, 1), stride=(3, 4))
comp_conv2d(conv2d, X).shape
```

下面，我们使用全连接层实现 1×1 卷积。 请注意，我们需要对输入和输出的数据形状进行调整。

```
def corr2d_multi_in_out_1x1(X, K):
    c_i, h, w = X.shape
    c_o = K.shape[0]
    X = X.reshape((c_i, h * w))
    K = K.reshape((c_o, c_i))
    Y = torch.matmul(K, X)  # 全连接层中的矩阵乘法
    return Y.reshape((c_o, h, w))
```