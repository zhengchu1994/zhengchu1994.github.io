---
title: Tensorflow2-AutoGraph
mathjax: true
date: 2020-10-01 21:53:32
tags: Tensorflow
categories: Tensorflow
visible:
---







### 一，Autograph编码规范总结

- 1，被`@tf.function`修饰的函数应尽可能使用TensorFlow中的函数而不是Python中的其他函数。例如使用`tf.print`而不是`print`，使用`tf.range`而不是`range`，使用`tf.constant(True)`而不是`True`.
- 2，避免在`@tf.function`修饰的函数内部定义`tf.Variable.`
- 3，被`@tf.function`修饰的函数不可修改该函数外部的Python列表或字典等数据结构变量。

### 二，Autograph编码规范解析

**1，被@tf.function修饰的函数应尽量使用TensorFlow中的函数而不是Python中的其他函数。**

```python
import numpy as np
import tensorflow as tf

@tf.function
def np_random():
    a = np.random.randn(3,3)
    tf.print(a)

@tf.function
def tf_random():
    a = tf.random.normal((3,3))
    tf.print(a)
```

```python
#np_random每次执行都是一样的结果。
np_random()
np_random()

array([[ 0.22619201, -0.4550123 , -0.42587565],
       [ 0.05429906,  0.2312667 , -1.44819738],
       [ 0.36571796,  1.45578986, -1.05348983]])
array([[ 0.22619201, -0.4550123 , -0.42587565],
       [ 0.05429906,  0.2312667 , -1.44819738],
       [ 0.36571796,  1.45578986, -1.05348983]])
```



```python
#tf_random每次执行都会有重新生成随机数。
tf_random()
tf_random()

[[-1.38956189 -0.394843668 0.420657277]
 [2.87235498 -1.33740318 -0.533843279]
 [0.918233037 0.118598573 -0.399486482]]
[[-0.858178258 1.67509317 0.511889517]
 [-0.545829177 -2.20118237 -0.968222201]
 [0.733958483 -0.61904633 0.77440238]]
```





**2，避免在@tf.function修饰的函数内部定义tf.Variable.**

```python
# 避免在@tf.function修饰的函数内部定义tf.Variable.

x = tf.Variable(1.0,dtype=tf.float32)
@tf.function
def outer_var():
    x.assign_add(1.0)
    tf.print(x)
    return(x)

outer_var() 
outer_var()
```



**3,被@tf.function修饰的函数不可修改该函数外部的Python列表或字典等结构类型变量。**





### 一，Autograph的机制原理







### 一，Autograph和tf.Module概述



TensorFlow提供了一个基类`tf.Module`，通过继承它构建子类，我们不仅可以获得以上的自然而然，而且可以非常方便地管理变量，还可以非常方便地管理它引用的其它Module，最重要的是，我们能够利用tf.saved_model保存模型并实现跨平台部署使用。

实际上，`tf.keras.models.Model`,`tf.keras.layers.Layer` 都是继承自`tf.Module`的，提供了方便的变量管理和所引用的子模块管理的功能。

**因此，利用`tf.Module`提供的封装，再结合TensoFlow丰富的低阶API，实际上我们能够基于TensorFlow开发任意机器学习模型(而非仅仅是神经网络模型)，并实现跨平台部署使用。**

```python
class DemoModule(tf.Module):
    def __init__(self,init_value = tf.constant(0.0),name=None):
        super(DemoModule, self).__init__(name=name)
        with self.name_scope:  #相当于with tf.name_scope("demo_module")
            self.x = tf.Variable(init_value,dtype = tf.float32,trainable=True)

		#在tf.function中用input_signature限定输入张量的签名类型：shape和dtype
    @tf.function(input_signature=[tf.TensorSpec(shape = [], dtype = tf.float32)])  
    def addprint(self,a):
        with self.name_scope:
            self.x.assign_add(a)
            tf.print(self.x)
            return(self.x)
          
          
#执行
demo = DemoModule(init_value = tf.constant(1.0))
result = demo.addprint(tf.constant(5.0))

#查看模块中的全部变量和全部可训练变量
print(demo.variables)
print(demo.trainable_variables)

#查看模块中的全部子模块
demo.submodules

#使用tf.saved_model 保存模型，并指定需要跨平台部署的方法
tf.saved_model.save(demo,"./data/demo/1",signatures = {"serving_default":demo.addprint})

#加载模型
demo2 = tf.saved_model.load("./data/demo/1")
demo2.addprint(tf.constant(5.0))





import datetime

# 创建日志
stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
logdir = './data/demomodule/%s' % stamp
writer = tf.summary.create_file_writer(logdir)

#开启autograph跟踪
tf.summary.trace_on(graph=True, profiler=True) 

#执行autograph
demo = DemoModule(init_value = tf.constant(0.0))
result = demo.addprint(tf.constant(5.0))

#将计算图信息写入日志
with writer.as_default():
    tf.summary.trace_export(
        name="demomodule",
        step=0,
        profiler_outdir=logdir)
```







# 五、TensorFlow的中阶API

TensorFlow的中阶API主要包括:

- 数据管道(tf.data)
- 特征列(tf.feature_column)
- 激活函数(tf.nn)
- 模型层(tf.keras.layers)
- 损失函数(tf.keras.losses)
- 评估函数(tf.keras.metrics)
- 优化器(tf.keras.optimizers)
- 回调函数(tf.keras.callbacks)





