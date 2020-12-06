---
title: PCA
mathjax: true
date: 2019-08-19 21:26:16
tags: MachineLearning
categories: 机器学习

---



### 主成分分析--两种视角



前言：降维方法：主要有主成分分析（PCA）、线性判别分析（LDA）、等距映射（Isomap）、局部线性嵌入（LLE）、拉普拉斯特征映射（LE）、局部保留投影（LPP）；



#### 最大方差

给定训练集合$\{x_n\},n=1,2,...,N$，样本维度是$D$

目标是：样本数据转换为投影数据，其维度为$M$，同时最大化投影数据的协方差。

 ✗ 最大化协方差是为何？

 ✔︎ **最大方差理论**：在信号处理中认为信号具有较大的方差，噪声有较小的方差，信噪比就是信号与噪声的方差比，越大越好。样本在横轴上的投影方差较大，在纵轴上的投影方差较小，那么认为纵轴上的投影是由噪声引起的。因此我们认为，最好的$k$维特征是将$n$维样本点转换为$k$维后，每一维上的样本方差都很大。



现在考虑投影数据到一维空间，设$D$维度的单位向量$\mathbf u_1$，（更关系该向量的方向而不是尺度），因此设$\mathbf u_1^T \mathbf u_1 =1$, 那么原来的数据$x_n$的投影为一个标量$\mathbf u_1^Tx_n$, 计算得到均值向量为$\mathbf u_1^T\bar x$, 其中：
$$
\bar x = \frac{1}{N}\sum_{n=1}^N=x_N
$$
计算协方差矩阵为:
$$
\frac{1}{N}\sum_{i=1}^N\{\mathbf u_1^Tx_i^T-\mathbf u_1^T\bar x \}^2= \mathbf u_1^T\frac{1}{N}\sum_{i=1}^N (x_i-\bar x)(x_i-\bar x)^T\mathbf u_1=
\mathbf u_1^T S\mathbf u_1
$$


$S$是样本协方差矩阵，下面成了解最优化方程如下：
$$
\mathbf u_1^TS \mathbf u_1 \\\\
s.t., \Vert \mathbf u_1\Vert = 1
$$


使用拉格朗日乘子，其中$\alpha\ge 1$：
$$
\mathbf u_1^TS \mathbf u_1  + \alpha_1(1-\mathbf u_1^T\mathbf u_1)
$$


计算导数前，先给出些公式；



#### 向量求导的几个公式

实值函数对向量求导的两个公式：（求导变量为$\mathbf x$）：
$$
\nabla A\mathbf {x}=A 
$$
向量内积的求导法则：

* 内积是一个实数，因此相当于实数对向量求导，结果是与自变量同型的向量。
* $\nabla(\mathbf a^T\mathbf x)= \mathbf a$
* $\nabla||\mathbf x||_2^2 =\nabla(\mathbf x^T\mathbf x)=2\mathbf x $
* $\nabla(\mathbf x^TA \mathbf x)=(A + A^T)\mathbf x$



由此，
$$
\nabla_{\mathbf u}(\mathbf u_1^TS \mathbf u_1  + \alpha_1(1-\mathbf u_1^T\mathbf u_1))= (S+S^T)\mathbf u -2\alpha_1\mathbf u =0 \\\\
S\mathbf u_1 =\alpha_1\mathbf u_1
$$


由此看出，$\mathbf u_1$ 是特征值$\alpha_1$的特征向量，而且还是最大特征值$\max \alpha_1$的特征向量。$\mathbf u_1$是第一个最大主成分，求取第二个最大主成分$\mathbf u_2$的公式是：
$$
\text{maximize} \ \mathbf u_2^2 S \mathbf u_2 \\\\
\text{subject to} \ \mathbf u_2^T \mathbf u_1 =0, ||\mathbf u_2||=1.
$$
拉格朗日形式：
$$
\mathbf u_2^TS\mathbf u_2 +\alpha_2(1-\mathbf u_2^T\mathbf u_2) +\beta \mathbf u_2^T \mathbf u_1
$$


最后得到 $\beta=0$,$Su_2=\alpha_2 u_2$,看出第二大主成分是$S$的第二大特征值。





* 葫芦百面的解释：给定一组数据点$\{v_1,v_2,...,v_n\}$,所有向量均为列向量，中心化后的表示为$\{x_1,x_2,...,x_n\}=\{v_1-\mu,v_2-\mu,...,v_n-\mu \}$，其中$\mu=\frac{1}{n}\sum_{i=1}^nv_i$。

  向量内积在几何上表示第一个向量投影到第二个向量上的长度，因此向量$x_i$投影到$w$上的投影坐标为$<x_i,w>=x_i^Tw$, 然后就是之前了解的PCA的目标：找到这么一个投影方向$w$，使得所有向量$x_i$在$w$上的投影方差尽可能的大。

   为什么要中心化？我从这里得到了答案，因为投影之后的均值仍为0：
  $$
  \mu^*=\frac{1}{n}\sum_{i=1}^nx_i^Tw=\left(\frac{1}{n}\sum_{i=1}^nx_i^T\right)w = 0
  $$
  

形式化了PCA的求解方法：

1. 对样本数据进行中心化处理；

2. 求样本协方差矩阵；

3. 对协方差矩阵进行特征值分解，将特征值从大到小排列；

4. 取前面$k$大的特征值对应的特征向量$w_1,w_2,...,w_k$,通过以下映射将$n$维样本向量映射到$k$维：
   $$
   X_i^* =
   \begin{bmatrix}
   w_1^Tx_i \\
   w_2^Tx_i \\
   \cdots\\
   w_k^Tx_i
   \end{bmatrix}
   $$
   

新的$x_i^*$的第$k$维度就是$x_i$在第$k$个主成分$w_k$方向上的投影，并且给出降维后的信息占比:
$$
\eta = \sqrt{\frac{\sum_{i=1}^k\lambda_i^2}{\sum_{i=1}^n\lambda_i^2}}
$$


（2）PCA 最小平方误差理论

先记录一个引出这个证明的线索：数据中每个点$x_k$到$d$维超平面$D$的距离为该点对应的向量与该点在超平面上的投影做减法，对应向量长度为$\text{distance}(x_k,D)= ||x_k - \tilde x_k||$ ,$\tilde x_k$表示$x_k$在超平面$D$上的投影向量。假设超平面由$d$个标准正交基$W=\{w_1,w_2,...,w_d\}$构成，我们有：
$$
\tilde x_k = \sum_{i=1}^d(w_i^Tx_k)w_i
$$
$w_i^Tx_k$表示$x_k$在$w_i$方向上的投影长度，因此$\tilde x_k$实际表示的就是$x_k$在$W$这组标准正交基下的坐标。因此，PCA的优化目标是：
$$
\arg \min_{w_1,w_2,...,w_d} =\sum_{k=1}^n ||x_k -\tilde x_k||_2^2 \\\\
s.t., w_i^T w_j = \delta_{ij} = \begin{cases}1,&i=j ,{\forall i,j} \\\\0, &i\neq j ,{\forall i,j}\end{cases}
$$


之后便是求最优解了。



























-----

* Reference:
  1. ：hulu百面中给的方法  https://mp.weixin.qq.com/s?__biz=MzA5NzQyNTcxMA==&mid=2656430435&idx=1&sn=f55f0ad0b5025076f8b9cbb248737f75&chksm=8b004922bc77c034b92a97387d791ac350f643130da03534dfadfe28a162c8fab98441e0a7b9&scene=21#wechat_redirect
  2. ：PCA 最小平方误差理论 https://zhuanlan.zhihu.com/p/33238895
  3.  证明PCA的最大方差第二项也同样方差最大化：
  4. https://www.cs.toronto.edu/~urtasun/courses/CSC411/tutorial8.pdf
  5. https://blog.csdn.net/han____shuai/article/details/50573066