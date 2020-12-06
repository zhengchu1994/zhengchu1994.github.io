---
title: 逻辑与线性回归
mathjax: true
date: 2020-06-16 16:07:11
tags: MachineLearning
categories: 机器学习

---



update：2020-06-16

last：2019-08-16

### 逻辑回归模型

一、定义：逻辑回归模型是用输入$x$的线性函数对输出$Y$的对数几率进行建模的模型，二项逻辑回归的形式为：
$$
\log\frac{P(Y=1|x)}{P(Y=0|x)}=w \cdot x
$$

* 其中$w=(w^{(1)},w^{(2)},\dots,w^{(n)},b)^T$ , $x=(x^{(1)},x^{(2)},\dots,x^{(n)},1)^T$.

*  其中，一个事件的几率是该事件发生与不发生的概率之比，设事件发生的概率是$p$，那么该事件的几率是$\frac{p}{1-p}$,对数几率是$\log\frac{p}{1-p}$.

二、最大似然估计法估计逻辑回归模型

设给定了训练集合$T={(x_1,y_1),(x_2,y_2),\dots,(x_N,y_N)}$, 其中$x_i\in\mathbf R^n$,$y_i\in\{0,1\}$, 设：
$$
P(Y=1|x) = \pi(x) , \  \ \  P(Y=0|x)=1-\pi(x)
$$
对数似然函数为(发生与不发生事件的**概率积**)： 
$$
\prod_{i=1}^N\left[\pi(x_i)^{y_i} \right]\left[1-\pi(x_i)^{1-y_i} \right]
$$


✔︎  逻辑回归与线性回归的异同?

* 异：
  * 1. 逻辑回归中，模型学习得出的是$E\left[y|x;\theta\right]$,即给定自变量和超参数后，学习因变量的期望；线性回归中，，模型求解的是$\hat y=\theta^T\cdot x$,是对我们假设的线性关系$ y=\theta^T\cdot x+\epsilon$的一个近似。 
  * 2. 最大的区别是，逻辑回归中的因变量是离散的，而线性回归中因变量是连续的。逻辑回归中$y$为因变量，而非对数几率值。
* 同：
  * 都是用极大似然函数对训练样本建模；
  * 都使用梯度下降法。



对数似然函数的推导：
$$
L(w)\\\\
=\sum_{i=1}^{N}[y_i\log\pi(x_i)+(1-y_i)\log(1-\pi(x_i))]\\\\
=\sum_{i=1}^N\left[y_i\log\frac{\pi(x_i)}{1-\pi(x_i)}+\log(1-\pi(x_i))\right] \\\\
=\sum_{i=1}^N\left[ y_i(w\cdot x_i) -\log(1+\exp(w\cdot x_i))\right]
$$


其中最后一步的右边项为：
$$
\begin{align}
\log\frac{\pi(x_i)}{(1-\pi(x_i))}= w\cdot x_i \\\\
\frac{\pi(x_i)}{(1-\pi(x_i))}= \exp(w\cdot x_i)  \\\\
\frac{1}{\exp(w\cdot x_i)} = \frac{(1-\pi(x_i))}{\pi(x_i)}= \frac{1}{\pi(x_i)}-1\\\\
\frac{1+\exp(w\cdot x_i)}{\exp(w\cdot x_i)} = \frac{1}{\pi(x_i)}\\\\
\pi(x_i) = \frac{\exp(w\cdot x_i)}{1+\exp(w\cdot x_i)} \\\\
1-\pi(x_i) = 1-\frac{\exp(w\cdot x_i)}{1+\exp(w\cdot x_i)} =\frac{1}{1+\exp(w\cdot x_i)} \\\\
\log(1-\pi(x_i)) = -\log(1+\exp(w\cdot x_i))
\end{align}
$$



最后便是梯度下降或者拟牛顿法求解最优化问题，得到目标参数$\hat w$.



三、多项逻辑回归公式

对应多分类模型，我们有$Y\in{1,2,\dots,K}$，$x\in\mathbf R^{n+1}$,$w_k\in\mathbf R^{n+1}$, 各个类别的分类概率为:
$$
P(Y=k|x)=\frac{\exp(w_k\cdot x)}{1+\sum_{k=1}^{K-1}\exp{(w_k \cdot x)}} , \  k=1,2,3,\dots,K-1  \\\\
P(Y=K|x)= \frac{1}{1+\sum_{k=1}^{K-1}\exp(w_k \cdot x)}
$$



 

### 最大熵原理与最大熵模型

 

（1）最大熵原理是概率模型的一个准则，即所有可能的概率模型（分布）中，最好的模型认为是熵最大的模型。看离散随机变量的熵公式：
$$
H(P)=-\sum_xP(x)\log P(x)\\\\
0\le H(P) \le\log\vert X\vert
$$
其中$|X|$是$X$的取值个数，如果$X$服从均匀分布，那么右边等式相等，也就是熵最大。

★ 该原理表达的思想很直接，没有新的信息到来的时候，其他不确定的部分都是“等可能的”，而度量可能性的方法便是最大化数据的熵。



（2）最大熵模型是最大熵原理在分类上的应用; 假设分类模型是一个条件概率模型$P(Y|X)$,训练集合为$T=\{(x_1,y_1),(x_2,y_2),\cdots,(x_N,y_N)\}$, 从训练集可以得到联合分布的经验分布$\hat P=(X,Y)$和$\hat P(X)$：
$$
\hat P=(X=x,Y=y) =\frac{\nu(X=x,Y=y)}{N} \\\\
\hat P(X) = \frac{\nu(X=x)}{N}
$$


这里$\nu(X=x,Y=y)$表示训练数据中样本$(x,y)$出现的频数，$\nu(X=x)$表示训练数据中输入$x$出现的频数，$N$是训练样本容量。



特征函数$f(x,y)$ 输出0和1来表示$x$和$y$是否满足某一事实.



★★: $f(x,y)$ 有$n$个，每一个对训练集合中的$x$分类为不同的$y$的准则，如下图：





该函数关于经验分布$\hat P=(X,Y)$的期望：
$$
\hat E_{\hat P}(f) =\sum_{x,y}\hat P(x, y) f(x,y)
$$


 ★ $E_{\hat P}(f)$中的下标$\hat P$ 是一种标记，代表经验期望；反之，$E_{P}(f)$代表模型期望；

该函数关于模型$P(Y|X)$与经验分布$\hat P(x)$的期望值：
$$
E_{P}(f)=\sum_{x,y}\hat P(x)P(Y|X)f(x,y)
$$



★ 一、模型期望使用经验分布$\hat P(x)$作为拟合；二、上式中模型$P(Y|X)$是未知的,因此联合两个式子求模型。若模型真的拟合了数据，由条件概率公式可知，应该有$\hat P(Y|X)=\frac{\hat P(x, y)}{\hat P(x)}≈P(Y|X)$.

假设模型获取了训练数据的信息，那么上面两个期望相等，即：
$$
E_{P}(f) =  \hat E_{\hat P}(f) \\\\
\sum_{x,y}\hat P(x)P(Y|X)f(x,y) = \sum_{x,y}\hat P(x, y) f(x,y)
$$





方面下面的叙述，假设一个模型集合$\mathcal C$,其中有$n$个$f$函数作为特征约束：
$$
\mathcal C  \{P\in\mathcal P|E_P(f_i)=\hat E_{\hat p}(f_i),i=1,2,\cdots,n\}  \\\\
$$


★ 满足上面约束条件的模型集合中,还未出现最大熵模型的身影。直到给出下面的最大熵模型公式：
$$
H^{*}(P) = \max_{P\in\mathcal C} H(P)=-\sum_{x,y}\hat P(x)P(y|x)\log P(y|x)
$$
其中左边是我写的，为了清楚看出熵要得到最大时，该公式才表达为最大熵模型。



（3）最大熵模型的学习是求解约束最优化问题，即：
$$
\max_{P\in\mathcal C} H(P)=-\sum_{x,y}\hat P(x)P(y|x)\ \log P(y|x)\\\\
\min_{P\in \mathcal C}-H(P)=\sum_{x,y}\hat P(x)P(y|x)\ logP(y|x) \\\\
s.t., E_{P}(f_i) - E_{\hat P}(f_i)=0, i=1,2,...,n, \\\\
\sum_{y} P(y|x) = 1
$$


引入拉格朗日函数求$(3)$的对偶问题的解：
$$
L(P,w)=-H(P) + w_0(1-\sum_{y} P(y|x) ) + \sum_{i=1}^nw_i(E_{\hat P}(f_i) - E_{P}(f_i))  \\\\
= \sum_{x,y}\hat P(x)P(y|x)\log P(y|x) + w_0(1-\sum_{y} P(y|x) ) + \\\\
\sum_{i=1}^nw_i(\sum_{x,y}\hat P(x,y)f_i(x,y) - \sum_{x,y}\hat P(x)P(y|x)f_i(x,y))
$$


原始问题是
$$
\min_{P\in \mathcal C}\max_{w}L(P,w)
$$
对偶问题为
$$
\max_{w}\min_{P\in \mathcal C}L(P,w)
$$





-------------



![算法题解-5](https://tva1.sinaimg.cn/large/007S8ZIlgy1gdrdjqe44gj31ad0u07q0.jpg)







### LinearRegression

$m$：训练集大小。
$$
\text{MSE}(\theta)=\frac{1}{m}(\theta^{T}x^{(i)} - y^{(i)})^2
$$


#### Ridge Regression：$L_2$ norm

注意：输入尺度的规范化。（`StandardScaler`）

$\theta_0$不正则化，也就是bias不正则化。
$$
J(\theta) = \text{MSE}(\theta) + \alpha\frac{1}{2}\sum_{i=1}^n\theta_{i}^2
$$

#### Lasso Regression：$L_1$ norm

$$
J(\theta)= \text{MSE}(\theta) +\alpha \frac{1}{2} \sum_{i=1}^n\vert\theta_i\vert 
$$



#### Logistic Regression

比如$h_{\theta}(\mathbb{x}) = \sigma(\theta_0 + \theta_1x_1 + \theta_2x_2)$，其中 $\mathbb{x}$ 是单个样本，$x_i$是其第$i$维度的特征。
$$
p(x)=h_{\theta}(x) = \sigma(x^T\theta) \\
\sigma(t) = \frac{1}{1 + e^{(-t)}}
$$
二分类里有$\hat p >= 0.5$ ，那么$\hat y = 1$。



LOSS：
$$
J(\theta) = -\frac{1}{m}\sum_{i=1}^m\bigg[y^{(i)}\log\big[\hat p^{(i)}\big] + (1 -y^{(i)})\log\big[1-p^{(i)} \big]   \bigg]
$$


#### 多分类softmax



其中，样本是$\mathbf x$，模型输出的$\text{logit}=s(\mathbf x)$，$\hat p_k$是样本$\mathbf x$在预测为第$k$类的概率。
$$
\hat p_k = \sigma(s(\mathbf x))_k= \frac{\exp(s_k(\mathbf x))}{\sum_{j=1}^K \exp(s_j(\mathbf x))}
$$




损失：
$$
J(\theta) = -\frac{1}{m}\sum_{i = 1}^m \sum_{k=1}^K y_k^{(i)}\log(\hat p_k^{(i)})
$$
其中，$y_k^{(i)}$ 是第$i$个样本属于类别$k$的概率，等于0或1。





### 归纳偏执（bias And Variance）



* Bias：模型假设错误，比如数据呈非线性，但用的模型是线性回归模型；带来的问题往往是**Underfit the training data**，也就是缺少泛化性。
* Variance：模型对训练数据中的小变化敏感，原因在于模型的**degrees of freedom**比较大，形成过拟合，**Overfit with training data**
* Irriducible error：**Noisiness of  data itself**

总之，更复杂的模型（increasing complexity of models）容易过拟合（ high-variance），简单的模型容易欠拟合（high-bias）。