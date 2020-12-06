---
title: EM
mathjax: true
date: 2019-09-05 21:20:29
tags: MachineLearning
categories: 机器学习
visible:
---



### EM算法

EM算法解决的问题：有隐变量存在的概率模型参数的极大似然估计，或极大后验估计。

 

先给出《统计学习方法》中对隐变量、观测变量的形式化如下：



$Y$表示观测随机变量（observed）的数据，$Z$ 表示隐随机变量（hidden）的数据；

观测数据$Y$的概率分布是$P(Y|\theta)$, 其中$\theta$是需要估计的模型参数, 对数似然函数为$L(\theta)= \log P(Y|\theta) $;

$Y$与$Z$的联合概率分布是$P(Y,Z|\theta)$, 对数似然函数为$\log P(Y,Z|\theta)$.



#### EM算法怎 么来的



 ★ 不管隐变量$Z$，我们平时用极大似然估计的时候就是写出极大似然函数，假设数据间独立同分布，并用对数形式方便计算，这里还是这样做，对隐变量$Z$，我们的做法是把它看为联合分布，即$Z$也是同 $Y$一起发生的，但是$Z$发生的具体过程我们不知道，也就是没有$Z$的实例，先通过全概率公式分解它：
$$
L(\theta) = \mathrm{log}P(Y|\theta) = \mathrm{log}P(Y,Z|\theta) \\\\
=\mathrm{log}(\sum_Z P(Z|\theta)P(Y|Z,\theta))
$$



有个上面的公式，考虑怎么计算它。EM的做法是这样，现在先假设迭代了$i$次，得到参数估计 $\theta^{(i)}$, 后面的新估计设为$\theta$, 它的似然估计为$L(\theta)$，迭代更新使得似然估计增大，因此有：


$$
\begin{align*}
L(\theta) - L(\theta^{(i)}) &=\log (\sum_{Z}P(Z|\theta)P(Y|Z,\theta)) - \log P(Y|\theta^{(i)}) \\\\
&= \log \bigg(\sum_Z P(Z|Y, \theta^{(i)}) \frac{P(Y|Z,\theta)P(Z|\theta)} {P(Z|Y,\theta^{(i)})}\bigg) -\log P(Y|\theta^{(i)})   \\\\
&\ge\sum_{Z}P(Z|Y,\theta^{(i)})\log\bigg( \frac{P(Y|Z,\theta)P(Z|\theta)} {P(Z|Y,\theta^{(i)})}\bigg) -\log P(Y|\theta^{(i)}) \\\\
&=\sum_{Z}P(Z|Y,\theta^{(i)})\log\bigg( \frac{P(Y|Z,\theta)P(Z|\theta)} {P(Z|Y,\theta^{(i)})}\bigg)-\sum_{Z}P(Z|Y,\theta^{(i)}) \log P(Y|\theta^{(i)}) \\\\
&=\sum_{Z}P(Z|Y,\theta^{(i)})\log\bigg( \frac{P(Y|Z,\theta)P(Z|\theta)} {P(Z|Y,\theta^{(i)})P(Y|\theta^{(i)})}\bigg)
\end{align*}
$$


其中用到了Jensen不等式：
$$
\log \sum_j \lambda_jy_j \ge \sum_j\lambda_j \log y_j, \\\\
\lambda_j \ge 0, \sum_{j} \lambda_j = 1.
$$
令：
$$
B(\theta,\theta^{(i)}) \equiv L(\theta^{(i)}) +\sum_{Z}P(Z|Y,\theta^{(i)})\log\bigg( \frac{P(Y|Z,\theta)P(Z|\theta)} {P(Z|Y,\theta^{(i)})P(Y|\theta^{(i)})}\bigg)
$$


则：
$$
L(\theta) \ge B(\theta,\theta^{(i)})
$$


这就是EM想做的，为了使得$L(\theta)$增大，我们可以增大它的下界$B(\theta,\theta^{(i)})$ , 则有（省去$\theta^{(i)}$带来的常数项）：



$$
\begin{align*}
\theta^{(i+1)} &= \arg \max_{\theta}B(\theta, \theta^{(i)}) \\\\
&= \arg \max_{\theta}\bigg(L(\theta^{(i)}) +\sum_{Z}P(Z|Y,\theta^{(i)})\log \frac{P(Y|Z,\theta)P(Z|\theta)} {P(Z|Y,\theta^{(i)})P(Y|\theta^{(i)})}\bigg) \\\\
&=\arg \max_{\theta}\bigg(\sum_{Z}P(Z|Y,\theta^{(i)})\log P(Y|Z,\theta)P(Z|\theta) \bigg) \\\\
&=\arg \max_{\theta}\bigg(\sum_{Z}P(Z|Y,\theta^{(i)})\log P(Y,Z|\theta) \bigg) \\\\
&=\arg \max_{\theta}Q(\theta, \theta^{(i)})
\end{align*}
$$

上式等价于EM算法的一次迭代，即求$Q$函数及其极大化。下面给出$Q$函数定义EM算法流程。



#### EM算法流程

* 选择参数的初值$\theta^{(0)}$,开始迭代：

* E步：记$\theta^{(i)}$为第$i$次迭代得到的参数，在第$i+1$次迭代时我们求出$Q$函数：
  $$
  \begin{align*}
  Q(\theta,\theta^{(i)}) &= E_{Z}\big[\log P(Y,Z|\theta)|Y,\theta^{(i)}\big] \\\\
  &=\sum_{Z}P(Z|Y,\theta^{(i)})\log P(Y,Z|\theta)
  \end{align*}
  $$
  
* M步：极大化$Q$函数得到$\theta$ :
  $$
  \theta^{(i+1)} = \arg \max_\theta Q(\theta, \theta^{(i)})
  $$



停止迭代的条件是参数间的差小于某一个正数$\epsilon_1$ 或者$Q$函数的差小于某一个正数$\epsilon_2$：
$$
\Vert \theta^{(i+1)} - \theta^{(i)}\Vert < \epsilon_1  , \ or\\\\
\Vert Q(\theta^{(i+1)} ,\theta^{(i)})  -Q(\theta^{(i)} ,\theta^{(i)})  \Vert < \epsilon_2
$$


其中$Q$函数的定义是给定观测数据$Y$和当前参数 $\theta^{( i)}$下未观测数据的条件概率分布$P(Z|Y,\theta^{(i)})$的期望.

 ★ 既然给定了$P(Z|Y,\theta^{(i)})$,也就是知道了第$i$轮的$Z$发生概率，它在下一轮$\theta$的更新中，作为联合分布的已知变量出现：$Q(\theta,\theta^{(i)}) = E_{Z}\big[\log P(Y,Z|\theta)|Y,\theta^{(i)}\big] $ ，也就化未知变量$Z$为已知变量的方法。