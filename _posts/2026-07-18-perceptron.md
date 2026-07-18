---
title: "感知机"
date: 2026-07-18
layout: single
permalink: /deep-learning/perceptron/
sidebar:
  nav: docs
toc: true
toc_sticky: true
categories:
  - 深度学习
mathjax: true
---

## 感知机

### 模型

给定输入 $\mathbf{x}$，权重 $\mathbf{w}$ 和偏移 $b$，感知机输出：

$$
\begin{aligned}
o&=\sigma(\langle \mathbf{w},\mathbf{x}\rangle+b)
\qquad
\sigma(x)=
\begin{cases}
1, & \text{if } x>0\\
0 \ or \ -1, & \text{otherwise}
\end{cases}
\end{aligned}
$$

### 训练

<p align="center">
  <img src="{{ '/images/屏幕截图 2026-07-09 224242.png' | relative_url }}" width="339">
</p>

<p align="center">
  <img src="{{ '/images/屏幕截图 2026-07-09 223916.png' | relative_url }}" width="339">
</p>

$$
\begin{aligned}
&\textbf{initialize } w = 0 \text{ and } b = 0\\
&\textbf{repeat}\\
&\quad \textbf{if } y_i(\langle w,x_i\rangle+b)\leq 0 \textbf{ then}\\
&\qquad w \leftarrow w+y_ix_i \quad and \quad b\leftarrow b+y_i\\
&\quad \textbf{end if}\\
&\textbf{until } \text{all classified correctly}
\end{aligned}
$$

等价于使用批量大小为 1 的梯度下降，并使用如下的损失函数：$\ell(y,x,w)=\max(0,-y\langle w,x\rangle)$ 。

> [!warning]
> 感知机不能拟合 XOR 函数，它只能产生线性分割面。

## 多层感知机

### 在网络中加入隐藏层

我们可以通过在网络中加入一个或多个隐藏层来**克服线性模型的限制**， 使其能处理更普遍的函数关系类型。这种架构通常称为*多层感知机*（multilayer perceptron），通常缩写为*MLP* 。

<p align="center">
  <img src="{{ '/images/Pasted image 20260710102425.png' | relative_url }}" width="339">
</p>

这个多层感知机有 4 个输入，3 个输出，其隐藏层包含 5 个隐藏单元。 输入层不涉及任何计算，因此使用此网络产生输出只需要实现隐藏层和输出层的计算。 因此，这个多层感知机中的层数为 2 。

### 单隐藏层 — 单分类

- 输入 $x \in \mathbb{R}^n$
- 隐藏层 $W_1 \in \mathbb{R}^{m\times n} \ , \ b_1 \in \mathbb{R}^{m}$
- 输出层 $w_2 \in \mathbb{R}^{m} \ , \ b_2 \in \mathbb{R}$

$$
\begin{aligned}
h &= \sigma(W_1x+b_1) \\
o &= w_2^Th+b_2
\end{aligned}
$$

$\sigma$ 是按元素的激活函数

### 单隐藏层 — 多类分类

- 输入 $\mathbf{x}\in\mathbb{R}^n$
- 隐藏层 $\mathbf{W}_1\in\mathbb{R}^{m\times n} \ , \ \mathbf{b}_1\in\mathbb{R}^m$
- 输出层 $\mathbf{W}_2\in\mathbb{R}^{m\times k} \ , \ \mathbf{b}_2\in\mathbb{R}^k$

$$
\begin{aligned}
\mathbf{h} &= \sigma(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1) \\
\mathbf{o} &= \mathbf{W}_2^{T}\mathbf{h}+\mathbf{b}_2 \\
\mathbf{y} &= \operatorname{softmax}(\mathbf{o})
\end{aligned}
$$

### 多隐藏层

<p align="center">
  <img src="{{ '/images/屏幕截图 2026-07-10 123758.png' | relative_url }}" width="339">
</p>

$$
\begin{aligned}
\mathbf{h}_1 &= \sigma(\mathbf{W}_1\mathbf{x}+\mathbf{b}_1) \\
\mathbf{h}_2 &= \sigma(\mathbf{W}_2\mathbf{h}_1+\mathbf{b}_2) \\
\mathbf{h}_3 &= \sigma(\mathbf{W}_3\mathbf{h}_2+\mathbf{b}_3) \\
\mathbf{o} &= \mathbf{W}_4\mathbf{h}_3+\mathbf{b}_4
\end{aligned}
$$

## 激活函数

> _激活函数_（activation function）通过计算加权和并加上偏置来确定神经元是否应该被激活， 它们将输入信号转换为输出的可微运算。 大多数激活函数都是非线性的。

### sigmoid 激活函数

对于一个定义域在 $\mathbb{R}$ 中的输入，*sigmoid 函数* 将输入变换为区间 $(0,1)$ 上的输出。因此，sigmoid 通常称为*挤压函数*（squashing function）：它将范围 $(-\infty,\infty)$ 中的任意输入压缩到区间 $(0,1)$ 中的某个值：

$$
\operatorname{sigmoid}(x)=\frac{1}{1+\exp(-x)}
$$

<p align="center">
  <img src="{{ '/images/Pasted image 20260710111108.png' | relative_url }}" width="339">
</p>

### tanh 函数

与 sigmoid 函数类似， tanh（双曲正切）函数也能将其输入压缩转换到区间 $(-1,1)$ 上。 tanh 函数的公式如下：

$$
\tanh(x)=\frac{1-\exp(-2x)}{1+\exp(-2x)}
$$

<p align="center">
  <img src="{{ '/images/Pasted image 20260710111706.png' | relative_url }}" width="339">
</p>

### ReLU 激活函数

最受欢迎的激活函数是*修正线性单元*（Rectified linear unit，_ReLU_ ）， 因为它实现简单，同时在各种预测任务中表现良好。

$$
\operatorname{ReLU}(x)=\max(0,x)
$$

<p align="center">
  <img src="{{ '/images/Pasted image 20260710112428.png' | relative_url }}" width="339">
</p>