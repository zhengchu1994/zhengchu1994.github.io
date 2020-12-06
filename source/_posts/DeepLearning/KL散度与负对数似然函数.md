---
title: KL散度与负对数似然函数
mathjax: true
date: 2020-06-13 21:07:10
tags: DeepLearning
categories: 深度学习
visible:
---





------------

[toc]



更新时间：20.06.16

上次时间：date: 2019-09-08 17:17:10



#### 最小负对数似然

先给出似然函数：
$$
\mathcal L(\theta | x_1,...,x_n) = f(x_1,x_2,...,x_n|\theta) = \prod_{i=1}^nf(x_i|n)
$$


为什么对上面的似然函数取对数的原因：

1、数值分析：因为似然都很好，是小数的积，$\log$可以降低计算时发生潜在的下溢；

2、积分方面：$\log$ 变换使得乘性计算变为加性计算，更加方便；

3、积分方面：$\log$ 是单调变换函数，自变量与因变量同时保持增减；

因此取对数似然函数：
$$
\mathcal L(\theta|x_1,...,x_n) = \log\prod_{i=1}^nf(x_i|n) =\sum_{i=1}^n\log f(x_i|n)
$$
最大似然估计值（maximum likelihood estimator）定义为：
$$
\hat \theta_{\mathrm{MLE}}=\arg \max_{\theta} \sum_{i=1}^n\log f(x_i | n)
$$
根据：
$$
\arg \max_{X}(X) = \arg \min_{X}(-X)
$$
得到负对数似然估计值  （NLL： negative log-likelihood）：


$$
\begin{align}
\hat \theta_{\mathrm{MLE}} &=\arg \max_{\theta} \sum_{i=1}^n \log f(x_i|\theta)      \\
&=\arg \min_{\theta} -\sum_{i=1}^n \log f(x_i|\theta)
\end{align}
$$


#### 最大似然 =  最小KL散度



KL散度的计算公式，我们有：
$$
\mathrm{KL}(P(x|\theta^{\ast}) \Vert P(x|\theta)) = \mathbb E_{x\sim P(x|\theta^{\ast})}\bigg[\log \frac{P(x|\theta^{\ast})}{P(x|\theta)} \bigg] \\\\
= \mathbb E_{x\sim P(x|\theta^{\ast})}\bigg[\log P(x|\theta^{\ast}) \bigg] - \mathbb E_{x\sim P(x|\theta^{\ast})}\bigg[\log P(x|\theta) \bigg]
$$
其中$\theta^{\ast}$是我们的真实数据分布的参数，$\theta$是我们对由训练数据得到的估计参数。

左边是分布$P(x|\theta^{\ast})$的熵，是与估计$\theta$无关的常量，因此可以忽略；

在分布 $P(x|\theta^{\ast})$的数据中采样$n$个样本，由强大数定理可知，当$n$趋于无穷的时候：
$$
-\frac{1}{n}\sum_{i=1}^n \log P(x_i|\theta) = -\mathbb E_{x\sim P(x|\theta^{\ast})}[\log P(x|\theta)] \\\\
= \mathrm{constant}*\mathrm{NLL}
$$


也就是说**最小化真实数据的分布与采样分布间的KL散度等于最小化负对数似然函数**。





----



#### EMBO



给定数据$X$，数据自身形成数据分布$p(X)$（称为**prior**），据此可以计算最大似然估计$p(X|z)$（即**likelihood**），其中$z$是需要估计的模型参数。

但更普遍的情况是，我们知道模型的参数服从分布$p(z)$，所以更合理的做法是考虑上参数的分布，据此可以计算最大后验估计$p(z|X)$（即**posterior**），有：
$$
p(z|X)= \frac{p(X|z)p(z)}{p(X)} = \frac{\text{Likelihood}\times \text{Parameter's prior}}{\text{Evidence}}
$$


但问题是$p(X)$一般求不了：(要求在可能的所有参数$z$上求导)
$$
p(X) = \int p(X|z)p(z)dz
$$
解决这个问题的办法：如变分推断（即**Variational Inference**），做法比较直观，取一个被$\lambda$参数化的分布族$q(z;\lambda)$ 去近似$p(z|X)$。

由此考虑优化参数$\lambda $，使得分布$q(z)$与$p(z|X)$之间的KL散度降低，记为$D_\text{KL}(q||p)$：
$$
\begin{align}
D_\text{KL}(q||p) &= \mathbb E_q \big[\log \frac{q(z)}{p(z|X)} \big] \\
&= \mathbb E_q \big[\log q(z) - \log p(z|X) \big] \\
&= \mathbb E_q[\log q(z) - \log \frac{p(z, X)}{ p(X)}] \\
&= \mathbb E_q[\log q(z) - \log p(z, X) + \log p(X)] \\ 
&= \mathbb E_q[\log p(X)] - \big(\mathbb E_q[\log p(z, X)] -\mathbb E_q [\log q(z)]\big) \\
&= \mathbb E_q[\log p(X)] - \text{ELBO}
\end{align}
$$


为了最小化$D_\text{KL}(q||p)$，等价于最大化$\text{ELBO}$。

怎么求$\text{ELBO}$？

方法一、在平均场变分推断中（即**Mean field variational inference**）的做法，选取一个分布族$q(z;\lambda)$和参数先验$q(z)$ 去简化 $\text{ELBO}$.

方法二、可以利用类似EM算法的迭代方式去做，该方法称为**黑盒随机变分推断**（即**Black box stochastic variational inference**），做法：每次从$q$分布中采样一些样本，然后估计$\text{ELBO}$函数中对于$\lambda$的的梯度，然后随机梯度上升最大化$\text{ELBO}$。



-----





-----



Reference：

https://wiseodd.github.io/techblog/2017/01/26/kl-mle/

https://quantivity.wordpress.com/2011/05/23/why-minimize-negative-log-likelihood/

github/ageron/handson-ml2