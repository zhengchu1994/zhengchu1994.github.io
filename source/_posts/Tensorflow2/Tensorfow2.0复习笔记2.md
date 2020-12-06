---
title: Tensorfow2复习--技巧
mathjax: true
date: 2020-06-15 19:00:00
tags: DeepLearning
categories: DeepLearning
visible:
---



# Tensorfow2.0复习笔记2



#### 梯度消失/爆炸

问题：Gradients often **get smaller and smaller** as the algorithm progresses **down to the lower layers.**  

原因：通过神经网络输出后的参数方差（Variance）都要大于输入时的参数方差，层数越多，在经过顶层的激活函数后最终满溢（**Saturation**）

* 比如input尺度越大，被sigmoid函数激活后，不是1就是0，其导数也是接近0，所以做梯度下降时会trivival。



* 解决方法之一：**Glorot and He Initialization：the authors argue that we need the variance of the outputs of each layer to be equal to the variance of its inputs**（理想情况下）；初始化为$\text{mean} = 0$，方差等于如下的权重：
  * $\text{fan}_{\text{in}}$ 与 $\text{fan}_{\text{out}}$  分别代表隐层的参数$W$的输入与输出大小：

$$
\text{fan}_{\text{avg}} = (\text{fan}_{\text{in}} + \text{fan}_{\text{out}}) / 2
$$



![image-20200729083706156](https://tva1.sinaimg.cn/large/007S8ZIlly1gh7jnhfr7bj31g60byabt.jpg)



——API：

```python
# 1.
keras.layers.Dense(10, activation="relu", kernel_initializer="he_normal")

#2.
init = keras.initializers.VarianceScaling(scale=2., mode='fan_avg',
                                          distribution='uniform')
keras.layers.Dense(10, activation="relu", kernel_initializer=init)
```





#### 激活函数的问题

* ReLU：问题：***dying ReLUs***: during training, some neurons effectively “die,” meaning they stop outputting anything other than 0.
* 解决方法：
  *  leaky ReLU：
     *  **优点**：避免神经元失活。【经验上：setting ***α* = 0.2** (a huge leak) seemed to result in better performance than *α* = 0.01 (a small leak). 】
  *  randomized leaky ReLU (RReLU)：
     * **优点**：如正则化器，避免训练数据时过拟合。
  *  *parametric leaky ReLU* (PReLU)：
     * **优点**：大量数据时不错；
     * **缺点**：小数据上对训练数据过拟合）
*  *exponential linear unit* (ELU) ：
     * **优点**：比ReLU及其变体都好
     * **缺点**：训练慢。
  * Scaled ELU (SELU)：
    * 优点：
      *  1、深的网络上表现好；
      *   2、self-normalize: the output of each layer will **tend to preserve a mean of 0 and standard deviation of 1 during training,** ；
      *  3、可用于CNNs。 缺点：不适用于RNNs、skip连接的网络。
    *  要求1: The input features must be **standardized** (mean 0 and standard deviation 1).
    *  要求2: Every hidden layer’s weights must be **initialized with LeCun normal initialization**.
  
* 经验：SELU > ELU > leaky ReLU (and its variants) > ReLU > tanh > logistic.





#### Batch-normalization问题

问题：上述策略是保存在training的初期没问题，但在training的过程中还是有梯度不稳定的情况。



解决方法：**Batch Normalization (BN)** ：an operation in the model just **before or after the activation function of each hidden layer**。

做法：evaluating the **mean** and **standard deviation** of the input over the current **mini-batch**：

​	* 每个样本计算一个均值和方差，然后对输入做normalization，然后参数化的平移。



![image-20200729083853198](https://tva1.sinaimg.cn/large/007S8ZIlly1gh7jpacz6ij311u0jw0us.jpg)



* Batch Normalization acts like a regularizer, reducing the need for other regularization techniques
* 