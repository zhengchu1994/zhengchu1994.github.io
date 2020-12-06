---
title: GMM
mathjax: true
date: 2020-06-13 19:26:00
tags: MachineLearning
categories: 机器学习
visible:
---







Freshing time：20.06.13

----------





为什么高斯混合模型要用隐变量？
$$
P(X) = \sum_{Z}p(X,Z) = \sum_{Z}p(X|Z)p(Z)
$$
★ 通常，通过$Z$是类别标签计算数据$X$的MLE，但是无监督情况下，不知道$Z$。因此我们假设$Z\sim\textit{Multinomial}(\phi)$也就是多项式分布（相当于$k$个面的骰子投$n$ 次的分布）作为假设的类别。

其中,$\phi_j \ge 0, \sum_{j=1}^k\phi_j = 1, \phi(j)$给出$p(z^{(i)} = j)$，其中每个样本$x^{(i)}$对应一个$z^{(i)}$.



设$x^{(i)} | z^{(i)} =j \sim N(\mu_j, \Sigma_j)$.


$$
L(\phi, \mu,\Sigma) = \sum_{i=1}^m\log P(x^{(i)};\phi,\mu,\Sigma)= \\\\
\sum_{i=1}^m\log \sum_{z^{(i)}=1}^K P(x^{(i)}| z^{(i)};\phi,\mu,\Sigma)*P(z^{(i)}; \phi)
$$
上式不可计算，但是如果知道了$z^{(i)}$就可以计算了：
$$
=\sum_{i=1}^m\log P(x^{(i)}| z^{(i)};\phi,\mu,\Sigma) + \log P(z^{(i)}; \phi)
$$
用EM算法进行计算：





* 单个样本的似然；

![image-20200505121349078](https://tva1.sinaimg.cn/large/007S8ZIlgy1gehg8ta8kij31520iqmzi.jpg)

* 全部样本的期望似然：
* ![image-20200505121447604](https://tva1.sinaimg.cn/large/007S8ZIlgy1gehg9qvmnyj315a0mawhq.jpg)



-----









* 两个圆形代表随机变量，$X^{(i)}$为 已知随机变量，白色的为隐变量；
* 两个变量之间的折线，代表$X\sim(\mu^{(i)},\Sigma^{(j)})$依赖于$Z^{(i)}=j$;
* 其他实线代表条件依赖。
* $m$个实例，$K$ 个高斯分布。



![image-20200613184317684](https://tva1.sinaimg.cn/large/007S8ZIlly1gfquo1rp1pj31d80jyjvl.jpg)



GMM初始的参数可以是Kmeans的参数：

```python
GaussianMixture(covariance_type='full', init_params='kmeans', max_iter=100,
                means_init=None, n_components=3, n_init=10,
                precisions_init=None, random_state=42, reg_covar=1e-06,
                tol=0.001, verbose=0, verbose_interval=10, warm_start=False,
                weights_init=None)
```







### 聚类数量未知：BIC

We cannot use the **inertia or the silhouette score** because they both assume that **the clusters are spherical**. Instead, we can try to find the model that minimizes a theoretical information criterion such as the **Bayesian Information Criterion (BIC)** or the **Akaike Information Criterion (AIC)**:



${BIC} = {\log(m)p - 2\log({\hat L})}$



${AIC} = 2p - 2\log(\hat L)$



*  $m$ is the number of instances.

*  $p$ is the number of parameters learned by the model.

* $\hat L$ is the maximized value of the likelihood function of the model. This is the conditional probability of the observed data $\mathbf{X}$, given the model and its optimized parameters.



Both BIC and AIC penalize models that have more parameters to learn (e.g., more clusters), and reward models that fit the data well (i.e., models that give a high likelihood to the observed data).

* 下图中的k是不同的聚类数：

![image-20200613185854429](https://tva1.sinaimg.cn/large/007S8ZIlly1gfqv48u3c3j30zk0dmwh3.jpg)

### Bayesian Gaussian Mixture Models

优势：对不必要的clusters赋予约等于0的权重。

做法：对聚类数目进行参数化。

```python
>>> from sklearn.mixture import BayesianGaussianMixture
>>> bgm = BayesianGaussianMixture(n_components=10, n_init=10)
>>> bgm.fit(X)
>>> np.round(bgm.weights_, 2)
array([0.4 , 0.21, 0.4 , 0. , 0. , 0. , 0. , 0. , 0. , 0. ])
```



![image-20200613200418213](https://tva1.sinaimg.cn/large/007S8ZIlly1gfqx0ah0qqj314e0gu77r.jpg)

