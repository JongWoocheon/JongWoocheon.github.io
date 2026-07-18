---
title: "softmax 回归"
layout: single
permalink: /deep-learning/softmax-regression/
sidebar:
  nav: docs
toc: true
toc_sticky: true
categories:
  - 深度学习
tags:
  - 线性神经网络
  - softmax 回归
  - 分类
mathjax: true
---

## 独热编码

我们对类别进行一位有效编码。

$$
\mathbf{y} = [y_1, y_2, \ldots, y_n]^T
$$

$$
y_i =
\begin{cases}
1, & \text{if } i = y \\
0, & \text{otherwise}
\end{cases}
$$

例如，$(1,0,0)$ 其中对应于“猫”、$(0,1,0)$ 对应于“鸡”、$(0,0,1)$ 对应于“狗”：

$$
y \in \{(1,0,0), (0,1,0), (0,0,1)\}
$$

## 网络架构

为了估计所有可能类别的条件概率，我们需要一个有多个输出的模型，每个类别对应一个输出。 为了解决线性模型的分类问题，我们需要和输出一样多的*仿射函数*（affine function）。下面我们为每个输入计算三个*未规范化的预测*（logit）：$o_1$、$o_2$和$o_3$。

$$
\begin{aligned}
o_1 &= x_1w_{11}+x_2w_{12}+x_3w_{13}+x_4w_{14}+b_1,\\
o_2 &= x_1w_{21}+x_2w_{22}+x_3w_{23}+x_4w_{24}+b_2,\\
o_3 &= x_1w_{31}+x_2w_{32}+x_3w_{33}+x_4w_{34}+b_3.
\end{aligned}
$$

<p align="center">
  <img src="{{ '/images/softmax-regression-network.png' | relative_url }}" width="350">
</p>

## 校验比例

将线性层的输出直接视为概率时存在一些问题： 一方面，我们没有限制这些输出数字的总和为1。 另一方面，根据输入的不同，它们可以为负值。

由此我们引入*softmax* 函数：

$$
\hat{\mathbf{y}}=\operatorname{softmax}(\mathbf{o})
\quad
\text{其中}
\quad
\hat{y}_j=\frac{\exp(o_j)}{\sum_k\exp(o_k)}
$$

## 交叉熵损失

*交叉熵*（Cross Entropy）是衡量两个概率分布之间差异的指标，常用于机器学习中评估模型预测结果与真实标签的接近程度，越小表示预测越准确。公式为 $l(\mathbf{y}, \hat{\mathbf{y}}) = -\sum_{k=1}^{K} y_k \log \hat{y}_k$ 。

### 负对数似然

① 似然函数

模型输出 $P(y\mid x;\theta)$，表示输入 $x$ 属于类别 $y$ 的概率。

那么整个数据集的[^1]似然函数是：

$$
L(\theta)=\prod_{i=1}^{n}P(y^{(i)}\mid x^{(i)};\theta)
$$

训练时希望这个概率尽可能大，即 $\max_{\theta} L(\theta)$ 。

② 取对数

$$
\log L(\theta)=\sum_{i=1}^{n}\log P(y^{(i)}\mid x^{(i)})
$$

③ 转换为最小化问题

最大似然希望“概率越大越好”，而机器学习优化器默认寻找“损失越小越好”，所以通过乘以 -1，把最大化问题转换为最小化问题。

$$
-\log L(\theta)=-\sum_{i=1}^{n}\log P(y^{(i)}\mid x^{(i)})
$$

### 为什么等于交叉熵

- 假设有 4 个类别
- 真实标签为 $y = (0, 0, 1, 0)$
- 模型预测概率为 $(0.1, 0.2, 0.6, 0.1)$

$$
\begin{aligned}
l 
&= -(0\log0.1+0\log0.2+1\log0.6+0\log0.1)\\
&= -\log0.6\\
&= -\log P(y=\text{第3类}\mid x)
\end{aligned}
$$

于是对于每一个样本都有：

$$
l(y^{(i)}, \hat{y}^{(i)}) = - \log P(y^{(i)} \mid x^{(i)})
$$

对所有样本求和就得到：

$$
\sum_{i=1}^{n}-\log P(y^{(i)}\mid x^{(i)})
=
\sum_{i=1}^{n}l(\mathbf{y}^{(i)},\hat{\mathbf{y}}^{(i)})
$$

### 导数

利用 softmax 函数的定义，我们得到：

$$
\begin{aligned}
l(\mathbf{y},\hat{\mathbf{y}})
&=-\sum_{j=1}^{q}y_j\log\frac{\exp(o_j)}
{\sum_{k=1}^{q}\exp(o_k)}\\
&=-\sum_{j=1}^{q}y_j\log\sum_{k=1}^{q}\exp(o_k)
-\sum_{j=1}^{q}y_jo_j\\
&=\log\sum_{k=1}^{q}\exp(o_k)
-\sum_{j=1}^{q}y_jo_j
\end{aligned}
$$

考虑相对于任何未规范化的预测 $o_j$ 的导数，我们得到：

$$
\partial_{o_j}l(\mathbf{y},\hat{\mathbf{y}})
=
\frac{\exp(o_j)}
{\sum_{k=1}^{q}\exp(o_k)}
-y_j
=
\operatorname{softmax}(\mathbf{o})_j-y_j 
=
\hat{y}_j - y_j
$$

这也是为什么现代深度学习框架会将 Softmax 与交叉熵合并实现：不仅数值更稳定，而且梯度表达式极其简单，只需要计算**预测概率减去真实标签**即可。

[^1]: 似然函数是在已知观测数据的条件下，用参数表示数据出现可能性的函数。
