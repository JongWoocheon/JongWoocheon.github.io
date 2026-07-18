---
title: "数值稳定性、模型初始化和激活函数"
date: 2026-07-18
layout: single
permalink: /deep-learning/numerical-stability/
sidebar:
  nav: docs
toc: true
toc_sticky: true
categories:
  - 深度学习
mathjax: true
---

## 神经网络的梯度

考虑一个具有 $L$ 层、输入 $\mathbf{x}$ 和输出 $\mathbf{o}$ 的深层网络。 每一层由变换 $f_l$ 定义， 该变换的参数为权重 $\mathbf{W}^{(l)}$， 其[^1]隐藏变量是 $\mathbf{h}^{(l)}$（令 $\mathbf{h}^{(0)} = \mathbf{x}$ ）。 我们的网络可以表示为：

$$
\mathbf{h}^{(l)} = f_l (\mathbf{h}^{(l-1)}) \qquad \text{ 因此 } \mathbf{o} = f_L \circ \ldots \circ f_1(\mathbf{x})
$$

如果所有隐藏变量和输入都是向量， 我们可以将 $\mathbf{o}$ 关于任何一组参数 $\mathbf{W}^{(l)}$ 的梯度写为下式：

$$
\partial_{\mathbf{W}^{(l)}} \mathbf{o} = \underbrace{\partial_{\mathbf{h}^{(L-1)}} \mathbf{h}^{(L)}}_{ \mathbf{M}^{(L)} \stackrel{\mathrm{def}}{=}} \cdot \ldots \cdot \underbrace{\partial_{\mathbf{h}^{(l)}} \mathbf{h}^{(l+1)}}_{ \mathbf{M}^{(l+1)} \stackrel{\mathrm{def}}{=}} \underbrace{\partial_{\mathbf{W}^{(l)}} \mathbf{h}^{(l)}}_{ \mathbf{v}^{(l)} \stackrel{\mathrm{def}}{=}}.
$$

相当与 $L-l$ 此矩阵乘法。 

### 梯度爆炸

引入如下的 MLP（为了简单省略偏移）：$\mathbf{h}^{t}=f_t(\mathbf{h}^{t-1})=\sigma(\mathbf{W}^t\mathbf{h}^{t-1}) \quad \sigma\text{ 是激活函数}$ 。

令 $\mathbf{z}=\mathbf{W}^t\mathbf{h}^{t-1}$，则 $\mathbf{h}^{t}=\sigma(\mathbf{z})=\begin{bmatrix}\sigma(z_1)\\\sigma(z_2)\\\sigma(z_3)\end{bmatrix}$，所以 $\frac{\partial \mathbf{h}^{t}}{\partial \mathbf{z}}=\operatorname{diag}(\sigma'(\mathbf{z}))=\begin{bmatrix}\sigma'(z_1)&0&\cdots\\0&\sigma'(z_2)&\cdots\\\vdots&\vdots&\ddots\end{bmatrix}$ 。

易得 $\frac{\partial \mathbf{h}^{t}}{\partial \mathbf{h}^{t-1}}=\frac{\partial \mathbf{h}^{t}}{\partial \mathbf{z}}\frac{\partial \mathbf{z}}{\partial \mathbf{h}^{t-1}}=\operatorname{diag}(\sigma'(\mathbf{W}^t\mathbf{h}^{t-1}))\mathbf{W}^t$ 。

$$
\prod_{i=t}^{d-1}
\frac{\partial \mathbf{h}^{t}}{\partial \mathbf{h}^{t-1}}
=
\prod_{i=t}^{d-1}
\operatorname{diag}(\sigma'(\mathbf{W}^t\mathbf{h}^{t-1}))\mathbf{W}^t
$$

如果 $d-t$ 很大，梯度的值将会很大。

> [!danger] 问题
> - 容易超出值域
> - 对学习率敏感

### 梯度消失

<p align="center">
  <img src="{{ '/images/Pasted image 20260716152212.png' | relative_url }}" width="350">
</p>

正如上图，当 sigmoid 函数的输入很大或是很小时，它的梯度都会消失。 此外，当反向传播通过许多层时，除非我们在刚刚好的地方， 这些地方 sigmoid 函数的输入接近于零，否则整个乘积的梯度可能会消失。

<aside class="callout callout--note" role="note">
  <p>我们要让训练更加稳定，即<strong>让梯度值在合理的范围内<strong>，有如下的方法：</p>
  <ul>
    <li>将乘法变加法：ResNet，LSTM</li>
    <li>梯度归一化，梯度裁剪</li>
    <li>合理的权重初始化和激活函数</li>
  </ul>
</aside>

## 参数初始化

我们可以将每层的输出和梯度都看作随机变量，让它们的均值和方差保持相对稳定。这样可以避免信号在网络层之间传播时发生爆炸或消失。

|                          正向                           |                                                                    反向                                                                     |
| :---------------------------------------------------: | :---------------------------------------------------------------------------------------------------------------------------------------: |
| $\mathbb{E}[h_i^t] = 0 \quad \mathrm{Var}[h_i^t] = a$ | $\mathbb{E}\left[\frac{\partial \ell}{\partial h_i^t}\right] = 0 \quad \mathrm{Var}\left[\frac{\partial \ell}{\partial h_i^t}\right] = b$ |

我们接着上面的 MLP 的例子继续，假设 $w_{i,j}^{t}$ 是[^2]独立同分布，$h_i^{t-1}$ 独立与  $w_{i,j}^{t}$（前一层的输出和下一层的权重没有关系）。那么 $\mathbb{E}[w_{i,j}^{t}]=0,\ \operatorname{Var}[w_{i,j}^{t}]=\gamma_t$ 。

假设没有激活函数，即 $\mathbf{h}^{t}=\mathbf{W}^t\mathbf{h}^{t-1}$，这里 $W^t\in \mathbb{R}^{n_t\times n_{t-1}}$ 。

$$
\mathbb{E}[h_i^t]
=
\mathbb{E}
\left[
\sum_j w_{i,j}^{t}h_j^{t-1}
\right]
=
\sum_j
\mathbb{E}[w_{i,j}^{t}]
\mathbb{E}[h_j^{t-1}]
=0
$$

$$
\begin{aligned}
\mathrm{Var}[h_i^t]
&= \mathbb{E}[(h_i^t)^2]-\mathbb{E}[h_i^t]^2 \\
&= \mathbb{E}\left[\left(\sum_j w_{i,j}^t h_j^{t-1}\right)^2\right] \\
&= \mathbb{E}\left[
\sum_j (w_{i,j}^t)^2(h_j^{t-1})^2
+
\sum_{j\neq k}w_{i,j}^tw_{i,k}^th_j^{t-1}h_k^{t-1}
\right] \\
&= \sum_j
\mathbb{E}\left[(w_{i,j}^t)^2\right]
\mathbb{E}\left[(h_j^{t-1})^2\right] \\
&= \sum_j
\mathrm{Var}[w_{i,j}^t]\mathrm{Var}[h_j^{t-1}] \\
&= n_{t-1}\gamma_t\mathrm{Var}[h_j^{t-1}]
\end{aligned}
$$

因为 $\mathrm{Var}[h_i^t]=\mathrm{Var}[h_j^{t-1}]$，所以 $n_{t-1}\gamma_t=1$ 。

根据反向的均值和方差，我们可以得到 $n_t\gamma_t=1$ 。

### Xavier 初始化

我们很难人为去控制输出的维度，所以使得 $n_{t-1}\gamma_t=1, \ n_t\gamma_t=1$ 是困难的。

为了**适配权重形状变换**，我们采取 Xavier 这种折中的初始化方式：

$$
\gamma_t(n_{t-1}+n_t)/2=1
\quad\rightarrow\quad
\gamma_t=\frac{2}{n_{t-1}+n_t}
$$

- 正态分布 $\mathcal{N}\left(0,\sqrt{\frac{2}{n_{t-1}+n_t}}\right)$ 
- 均匀分布 $\mathcal{U}\left(-\sqrt{\frac{6}{n_{t-1}+n_t}},\sqrt{\frac{6}{n_{t-1}+n_t}}\right)$

[^1]: 隐藏变量（hidden variable）指的是网络中间层产生的特征。

[^2]: 独立同分布（Independent and Identically Distributed，简称 **i.i.d.**）指一组随机变量**彼此之间互不影响**，并且**都来自同一个概率分布**。
