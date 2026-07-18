---
title: "权重衰减和暂退法"
date: 2026-07-18
layout: single
permalink: /deep-learning/weight-decay-and-dropout/
sidebar:
  nav: docs
toc: true
toc_sticky: true
categories:
  - 深度学习
mathjax: true
---

## 权重衰减

### 正则化

我们可以通过收集更多的训练数据来缓解过拟合， 但这可能成本很高，耗时颇多。因此我们引入[^1]正则化技术。

在训练参数化机器学习模型时， _权重衰减_（weight decay）是最广泛使用的正则化的技术之一， 它通常也被称为 $L_2$ *正则化* 。 这项技术通过函数与零的距离来衡量函数的复杂度， 因为在所有函数 $f$ 中，函数 $f=0$ （所有输入都得到值 $0$ ） 在某种意义上是最简单的。

### 使用均方范数作为硬性限制

- 通过限制参数值的选择范围来控制模型容量

$$
\min \ \ell(w,b) \quad \text{s.t.} \quad \|w\|^2 \leq \theta
$$

- 通常不限制偏移 $b$（基本无影响）
- 更小的 $\theta$ 意味着更强的正则项

### 使用均方范数作为柔性限制

- 对每个 $\theta$，都可以找到 $\lambda$ 使得之前的目标函数等价于：

$$
\min \ \ell(w,b)+\frac{\lambda}{2}\|w\|^2
$$

- 超参数 $\lambda$ 控制了正则项的重要程度：$\lambda = 0$ （无作用）；$\lambda \rightarrow \infty, \ w^* \rightarrow 0$

<p align="center">
  <img src="{{ '/images/屏幕截图 2026-07-11 182239.png' | relative_url }}" width="339">
</p>

### 梯度

$$
\frac{\partial}{\partial w}
\left(
\ell(w,b)+\frac{\lambda}{2}\|w\|^2
\right)
=
\frac{\partial \ell(w,b)}{\partial w}
+\lambda w
$$

$$
w_{t+1}
=
(1-\eta\lambda)w_t
-
\eta
\frac{\partial \ell(w_t,b_t)}{\partial w_t}
$$

> 通常 $\eta\lambda<1$，在深度学习中通常叫做权重衰退。

## 暂退法（ Dropout ）

在前向传播过程中，计算每一内部层的同时注入噪声，即在训练过程中**随机失活**一些神经元，让网络不要过度依赖某些特定神经元，从而提高泛化能力。这个想法被称为*暂退法* 。

### 无偏差的加入噪音

- 对 $\mathbf{x}$ 加入噪音得到 $\mathbf{x}'$，我们希望保持输出的期望不变： $E \, [\, \mathbf{x}' \, ]=\mathbf{x}$
- 暂退法对每个元素进行如下扰动：

$$
x_i'=
\begin{cases}
0 & \text{概率为 } p = \text{dropout rate} \\
\frac{x_i}{1-p} & \text{其他情况}
\end{cases}
$$

- 即该神经元有 $p$ 的概率会被丢弃，$1-p$ 的概率会被保留

### 训练中使用暂退法

我们通常将暂退法作用在隐藏全连接层的输出上。

<p align="center">
  <img src="{{ '/images/Pasted image 20260711213515.png' | relative_url }}">
</p>

$$
\begin{aligned}
\mathbf{h} &= \sigma(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1)\\
\mathbf{h}' &= \operatorname{dropout}(\mathbf{h})\\
\mathbf{o} &= \mathbf{W}_2\mathbf{h}'+\mathbf{b}_2\\
\mathbf{y} &= \operatorname{softmax}(\mathbf{o})
\end{aligned}
$$

> [!warning]
> 训练结束后，每个参数 $w$ 已经是固定数值 。推理时只做前向传播，不计算损失，自然不需要正则化。

[^1]: 正则化（regularization）是指为解决过拟合而加入额外信息的过程，用于限制模型复杂度、减少过拟合、提高泛化能力。
