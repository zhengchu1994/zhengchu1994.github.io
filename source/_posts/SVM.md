---
title: SVM
mathjax: true
date: 2019-08-13 14:34:25
tags: MachineLearning
categories: 机器学习
---



## 前言：拉格朗日对偶性

1. 假设$f(x),c_i(x),h_j(x)$是定义在$\mathbf R^n$ 上的连续可微函数。原始问题定义为满足下列条件的约束不等式：

$$
\min_{x\in\mathbf R^n}f(x),\\\\
\text{s.t.},
\begin{cases}
c_i(x)\le 0, i=1,2,...,k \\\\
h_j(x) = 0, j=1,2,...,l
\end{cases}
$$



通过拉格朗日乘子$\alpha_i,\beta_j$（其中$\alpha_i\ge0$），得到拉格朗日函数：
$$
L(x,\alpha,\beta)=f(x)+\sum_{i=1}^k\alpha_ic_i(x) +\sum_{j=1}^l\beta_jh_j(x)  
$$
把公式$( 1 )$看作是$x$的函数，那么在$x$满足条件的时候，最大化$L(x,\alpha,\beta)$得到的最优解必然是$f(x)$本身，现在设：
$$
\theta_P(x)=\max_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta)=f(x),\\\\
\text{x satisify constraint in equation (1)}
$$


现在极小化上述公式得到：
$$
\min_x\theta_P(x,\alpha,\beta)=\min_x\max_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta) =\min_x f(x) \\\\
\text{s.t. conditions in equation (1)}
$$


原始问题$( 1 )$转变为广义拉格朗日函数问题$( 4 )$，定义原始问题的解为：$p^{\ast}=\theta_P(x)$.



2.对偶问题的定义是：
$$
\max_{\alpha,\beta,\alpha_i\ge0}\theta_D(\alpha,\beta)=\max_{\alpha,\beta,\alpha_i\ge0}\min_xL(x,\alpha,\beta) \\\\
\text{s.t.}, \alpha_i \ge 0, i =1,2,...,k
$$
定义其解为 $d^{\ast}=\max_{\alpha,\beta,\alpha_i\ge0}\theta_D(\alpha,\beta)$.



3.原始问题与对偶问题的关系通过KKT联系



定理：若原始问题与对偶问题都有最优值，有：
$$
d^{\ast} =\max_{\alpha,\beta,\alpha_i\ge0}\min_xL(x,\alpha,\beta)\le \min_x\max_{\alpha,\beta:\alpha_i\ge0}L(x,\alpha,\beta)=p^{\ast}
$$



推论C.1: 设  $x^{\ast}$  和  $\alpha^{\ast}$ , $\beta^{\ast}$ 分别是原始问题$( 4 )$和对偶问题$( 5 )$的可行解,  当 $d^{\ast}=p^{\ast}$ 时， $x^{\ast}$ 和 $\alpha^{\ast}$,$ \beta^{\ast}$分别是原始问题( 4 )和对偶问题( 5 )的最优解。

定理C.2：假设函数$f(x)$和$c_i(x)$是凸函数，$h_j(x)$是仿射函数；并且假设不等式约束$c_i(x)$是严格可行的，即存在$x$对所有$i$有$c_i(x)<0$,则存在$x^{\ast}$和$\alpha^{\ast},\beta^{\ast}$，使得$x^*$是原始问题的解，$\alpha^{\ast},\beta^{\ast}$是对偶问题的解，并且：
$$
p^{\ast} = d^{\ast} =L(x^{\ast}, \alpha^{\ast} , \beta^{\ast})
$$


更方便我们去对比两者是否相同等价的方法是如下的KTT条件：

定理C.3：函数$f(x)$ 和$c_i(x)$ 是凸函数，$h_j(x)$  是仿射函数, 且存在$x$,对所有$i$有$c_i(x)<0$, 则使得$x^{\ast}$是原始问题的解，$\alpha^{\ast},\beta^{\ast}$是对偶问题的解的充要条件是$ x^{\ast}$和 $\alpha^{\ast},\beta^{\ast}$,满足如下KKT条件：
$$
\begin{cases}
\nabla_xL(x^{\ast},\alpha^{\ast},\beta^{\ast})=0 \\\\
\alpha_i^{\ast}c_i(x^{\ast}) = 0,\ i=1,2,\dots,k \\\\
c_i(x^{\ast}) \le 0,\ i=1,2,\dots,k \\\\
\alpha_i \ge 0, \ i=1,2,\dots,k \\\\
h_j(x^{\ast}) = 0, \ j=1,2,\dots,l
\end{cases}
$$

 此时有：$\alpha_i^{\ast}c_i(x^{\ast}) = 0$是KKT的对偶互补条件，可知当$\alpha_i^{\ast} \ge 0$ 时，有$c_i(x^{\ast})=0$。



★ 也就是说原始问题若能构造称为拉格朗日函数，且满足对偶互补条件，那么原始问题和对偶问题等价。



#### KKT条件from周志华《线性代数》

他的博客里从等式约束问题开始，再讲不等式约束问题，这里我直接把对不等式约束问题的理解记下。

不等式约束问题如下：
$$
\min f(x), \\\\
\text{s.t.,} g(x)\le 0
$$
约束不等式$ g(x)\le 0$称为原始可行性，据此定义可行域为$K=x\in\mathbb R^n | g(x)\le0$. 

假设满足约束的最佳解为$x^{\ast}$,有两种情况：

(1) $g(x^{\ast})<0$，最佳解位于$K$的内部，称为内部解，此时约束条件属于无效的；即驻点$x^{\ast}$满足$\nabla f=0$且$\lambda=0$。

(2) $g(x^{\ast})=0$，最佳解位于$K$的边界，即称为边界解,，此时约束条件是有效的；这时可以证明驻点$x^{\ast}$发生在$\nabla f \in \text{span}\nabla g$,也就是$\nabla f$ 可以被$\nabla g$ 线性表示，有$\nabla f = - \lambda \nabla g$。 这里分析下$\lambda$的正负：

希望最小化$f$ ，梯度$\nabla f$ 应该指向可行域$K$的内部；

★ 最优解最小值是在边界处得到，这时$g(x^{\ast})=0$, 因此指向边界的方向是梯度$\nabla f$ 的反方向；

$\nabla g$ 指向$K$的外部； 

★ 因为$g(x)$是小于等于0的函数，可行解要求最大化$g(x)$, 梯度$\nabla g$ 的方向是最大化$g(x)$的方向，也就是在$K$的外部（$g(x)>0$的区域）。

★ 因此， 由 $\nabla f\le 0, \nabla g \ge 0$,  $\nabla f = - \lambda \nabla g$，可知$\lambda \ge 0$ ; 当我们求解的是最大化$f(x)$的时候，$\nabla f \ge 0$, 此时可知$\lambda <=0$。



结论：由上可知，$\lambda g(x)= 0$ 恒成立，称为互补松弛性。该结果可推广至多个约束等式与约束不等式的情况。

★ 这个互补松弛性便是刚说的对偶互补条件。



-----

##  线性可分支持向量机

 假设给定训练数据集$T={(x_1,y_1),(x_2,y_2),...,(x_N,y_N)}$是线性可分的, 其中标签属于$y_i\in\mathcal Y ={+1,-1}$，是一个二分类问题，我们的目的是找到一个分离超平面 $ w^{\ast} \cdot x + b^{\ast}$= 0 ,使得正负样本正确分类，学习到的模型$f(x)=\operatorname{sign}(w^{\ast}\cdot x + b^{\ast}) $ 称为线性可分类支持向量机。

每一个训练数据的点作为分类预测的确信程度是不一样的, $y(w\cdot x +b)$正好表达了分类的确信程度与分类是否正确，为此引入了函数间隔与几何间隔：

* 函数间隔：$T$中每一个样本在超平面$(w,b)$上函数间隔是 
  $$
  \hat \gamma_{i} = y_i(w\cdot x_i+b)
  $$
  

  定义整个训练集$T$在超平面$(w,b)$上的函数间隔为:
  $$
  \hat \gamma =\mathrm{min_{i=1,..,N}}\hat \gamma_i
  $$
  

* 函数间隔受制于超平面$\{w,b\}$的伸缩性，不能反映固定训练数据中的点到超平面的距离，为此定义几何间隔为
  $$
  \gamma_i=y_i(\frac{w}{||w||}\cdot x_i + \frac{b}{||w||})
  $$
  

   其中$||w||$为$w$的$L_{2}$ 范数；定义整个训练集$T$在超平面$(w,b)$ 上的几何间隔为 :
  $$
  \gamma =\text{min}_{i=1,..,N} \gamma_i
  $$
  

* 有函数间隔与几何间隔的关系：$\gamma=\frac{\hat \gamma}{||w||}$，假设法向量$w$的规范化等于$||w||=1$，那么几何间隔等于函数间隔。





最大几何间隔：得到的几何间隔是数据点到超平面的距离，数据点若离超平面越来越近，那么它的类别越来越接近超平面另一侧的类别，  分类预测的确信程度越来越低，由此，我们要做的是最大化数据集在超平面上的几何间隔，形式化如下：
$$
\max_{w,b}\gamma \\
s.t. \  y_i(\frac{w}{||w||}\cdot x_i+\frac{b}{||w||})\ge \gamma,i=1,2,....,N \tag{1}
$$


约束每个数据点的最大几何间隔至少大于最近数据点到超平面的距离。上述公式等价于：
$$
\begin{cases}
\max_{w,b}\frac{\hat \gamma}{||w||} \\  \tag{2}
s.t., \  y_i(w\cdot x_i+b)\ge \hat\gamma,i=1,2,....,N
\end{cases}
$$
由于函数间隔的伸缩性，我们假设$\hat \gamma=1$,也就是说，同时最大化$\frac{1}{||w||}$等价于最小化$\frac{1}{2}||w||^2$,上式变为：
$$
\begin{cases}
\max_{w,b}\frac{1}{2}||w||^2 \\ \tag{3}
s.t. \  y_i(w\cdot x_i+b)-1\ge 0,i=1,2,....,N
\end{cases}
$$
上述满足凸二次规划问题，解上述约束最优化问题得到最优解$w^{\ast},b^{\ast}$,由此得到分离超平面$w^{\ast}\cdot x+b=0$,分类决策函数$f(x)=\operatorname{sign} (w^{\ast}\cdot x+b^{\ast})$



##### 支持向量

基于几何间隔的定义，并且函数间隔的设为1，那么支持向量对于公式$(3)$满足等号，即$w \cdot x_i + b = \pm 1$，到超平面的距离为$\frac{1}{\Vert w \Vert}$, 中间的长带宽度为$\frac{2}{\Vert w \Vert}$。



定理：最大间隔分离超平面存在且唯一。

* 其中，在证明存在性时，设最优解为$(w^{\ast},b^{\ast})$ ，由于数据集中包含正负样本点，因此$(w,b)=(0,b)$不是最优的可行解；

  ★ 因为分类决策函数是$f(x)=\text{sign}(w^{\ast}\cdot x + b) $, 若$w^{\ast}=0$，那么$f(x)$只能为正或者负。

 ★ 根据林轩田的笔记，若记超平面外的一点$x$ 到超平面$w^Tx + b =0$的距离为$\text{distance}(x, b, w)$.看怎么计算的。设超平面上有两个点$x^1$和$x^2$，同时满足$w^Tx^1 +b=0$ 和$w^Tx^2 +b=0$，相减得$w^T(x^2 - x^1) =0$；也就是说$(x^2-x^1)$是平面上任一向量，$w$是法向量。先在回到求解$\text{distance}(x, b, w)$，只要把$x-x^1$投影到$w$方向上，通过三角不等式得到结果便可，记他们之间的夹角为$\theta$,有：
$$
\text{distance}(x, b, w) = |(x - x^1)\cos(\theta)| = \vert\Vert x -x^1\Vert \cdot \frac{(x-x^1)w}{\Vert x-x^1\Vert \Vert w\Vert} \vert \\\\
= \frac{1}{\Vert w\Vert}|w^Tx -w^Tx^1| =\frac{1}{\Vert w\Vert}|w^Tx +b|
$$






### 对偶算法 

线性可分支持向量机的最优化问题$(3)$可以应用拉格朗日对偶性，转变为求其对偶问题（dual problem）。

首先构建拉格朗日函数：
$$
L(w,b,\alpha) = \frac{1}{2}\Vert w\Vert^2 - \sum_{i=1}^N\alpha_iy_i(w\cdot x_i+b) +\sum_{i=1}^N\alpha_i  \tag{4}
$$
其中，$\alpha_i\ge0，i=1,2,...,N$ 是拉格朗日乘子，原始问题的对偶问题是**极大极小问题**：
$$
\max_{\alpha}\min_{w,b}L(w, b, \alpha) \tag{5}
$$
首先求右边 $\min_{w,b}L(w, b, \alpha)$,对$w,b$求偏导数并等于$0$：
$$
\nabla_wL(w,b,\alpha)=w-\sum_{i=1}^N\alpha_iy_ix_i=0 \\ \tag{7}
\Rightarrow w= \sum_{i=1}^N\alpha_iy_ix_i
$$

$$
\nabla_bL(w,b,\alpha)=-\sum_{i=1}^N\alpha_iy_i=0 \\ \tag{8}
\Rightarrow \sum_{i=1}^N\alpha_iy_i=0
$$

带入$（7，8）$到$(4)$结果是：
$$
L(w,b,\alpha) = \frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_ix_j)- \\\\
\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_ix_j)+\sum_{i=1}^N\alpha_i \\\\ \tag{9}
= -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_ix_j)+\sum_{i=1}^N\alpha_i
$$
对式子$(9)$做$\max_{\alpha}L(w,b,\alpha)$:
$$
\max_{\alpha} -\frac{1}{2}\sum_{i=1}^N\sum_{j=1}^N\alpha_i\alpha_jy_iy_j(x_ix_j)+\sum_{i=i}^N\alpha_i  \\\\ \tag{10}
s.t. \ \sum_{i=1}^N\alpha_iy_i=0,\\\\
\alpha_i\ge0, i=1,2,\dots,N
$$



#### 原始问题与对偶问题是否是同一解

原始问题满足C.2的条件，主要考虑的是存在$x$使得所有小于等于0的约束不等式都小于0，对于$y_i(w\cdot x_i+b)-1\ge 0,i=1,2,....,N $是肯定满足的；由此，对偶问题与原始问题的解是相等的。

可以通过求解对偶问题来求解原始问题。

定理：设对偶问题的解$\alpha^{\ast}={\alpha_1^{\ast},\alpha_2^{\ast},...,\alpha_l^{\ast}}$是对偶最优化问题的解，则存在下标$j$使得$\alpha_j^{\ast} > 0$,（对偶松弛条件）并且如下可求解$w^{\ast}$和$b^{\ast}$：
$$
w^{\ast} =\sum_{i=1}^{N} \alpha_i^{\ast}y_i x_i \\\\
b^{\ast} = y_j - \sum_{i=1}^N\alpha_i^{\ast}y_i(x_i \cdot x_j)
$$




★ 因为原始问题与对偶问题同解，所以满足KKT条件，可以拿来计算原始问题的解。





## 线性支持向量机



目的是为了应对线性不可分的数据集，对每一个样本点都引入一个松弛变量$\xi_i$,约束条件为:
$$
y_i(w_i\cdot x + b)\ge 1- \xi_i
$$




## 核函数
## 非线性支持向量机