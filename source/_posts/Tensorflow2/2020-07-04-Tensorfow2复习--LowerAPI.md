---

title: Tensorfow2复习--LowerAPI
mathjax: true
date: 2020-07-04 19:00:00
tags: DeepLearning
categories: DeepLearning
visible:



---



# Tensorfow2.0--LowerAPI

![image-20200704184231346](https://tva1.sinaimg.cn/large/007S8ZIlly1ggf4nplhfnj31700u07hf.jpg)





![image-20200704184431104](https://tva1.sinaimg.cn/large/007S8ZIlly1ggf4pqf5g2j312q0hg77u.jpg)





#### numpy to tensorflow

Notice that NumPy uses **64-bit** precision **by default**, while Tensor‐ Flow uses 32-bit. 

This is because 32-bit precision is generally more than enough for neural networks, plus it runs faster and uses less RAM. 

So when you create a tensor from a NumPy array, make sure to set **dtype=tf.float32**.







#### Customizing Models and Training Algorithms



* Custom Loss Functions

```python
def huber_fn(y_true, y_pred):
    error = y_true - y_pred
    is_small_error = tf.abs(error) < 1
    squared_loss = tf.square(error) / 2
    linear_loss = tf.abs(error) - 0.5
    return tf.where(is_small_error, squared_loss, linear_loss)
```

```python
input_shape = X_train.shape[1:]

model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1),
])

model.compile(loss=huber_fn, optimizer="nadam", metrics=["mae"])
model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
```



#### Saving and Loading Models That Contain Custom Components

* 需求：对于自定义的损失函数，需要把阈值threshold设为超参数，需要继承`keras.losses.Loss`类，并重载`get_config()`方法：

- The constructor accepts **kwargs** and passes them **to the parent constructor**, which handles standard hyperparameters: the name of the loss and the reduction algorithm to use to aggregate the individual instance losses. 

  By default, it is **"sum_over_batch_size"**, which means that the loss will be the sum of the instance losses, weighted by the sample weights, if any, and divided by the batch size

- The **call()** method takes the labels and predictions, computes all the instance losses, and returns them.

- The **get_config()** method returns a dictionary mapping each hyperparameter name to its value

```python
class HuberLoss(keras.losses.Loss):
    def __init__(self, threshold=1.0, **kwargs):
        self.threshold = threshold
        super().__init__(**kwargs) #基类实现
    def call(self, y_true, y_pred):
        error = y_true - y_pred
        is_small_error = tf.abs(error) < self.threshold
        squared_loss = tf.square(error) / 2
        linear_loss  = self.threshold * tf.abs(error) - self.threshold**2 / 2
        return tf.where(is_small_error, squared_loss, linear_loss)
    def get_config(self):
        base_config = super().get_config() #由基类实现
        return {**base_config, "threshold": self.threshold}
```

```python
model = keras.models.Sequential([
    keras.layers.Dense(30, activation="selu", kernel_initializer="lecun_normal",
                       input_shape=input_shape),
    keras.layers.Dense(1),
])

model.compile(loss=HuberLoss(2.), optimizer="nadam", metrics=["mae"])

model.fit(X_train_scaled, y_train, epochs=2,
          validation_data=(X_valid_scaled, y_valid))
          
```



* Save & load：

```python
model.save("my_model_with_a_custom_loss_class.h5")
model = keras.models.load_model("my_model_with_a_custom_loss_class.h5", 
                                custom_objects={"HuberLoss": HuberLoss})
```





#### Custom Activation Functions, Initializers, Regularizers, and Constraints

```python
def my_softplus(z): # return value is just tf.nn.softplus(z)
    return tf.math.log(tf.exp(z) + 1.0)

def my_glorot_initializer(shape, dtype=tf.float32):
    stddev = tf.sqrt(2. / (shape[0] + shape[1]))
    return tf.random.normal(shape, stddev=stddev, dtype=dtype)

def my_l1_regularizer(weights):
    return tf.reduce_sum(tf.abs(0.01 * weights))

def my_positive_weights(weights): # return value is just tf.nn.relu(weights)
    return tf.where(weights < 0., tf.zeros_like(weights), weights)
```



* 继承实现：
  *  **keras.regularizers.Regularizer**, 
  * **keras.constraints.Constraint**, 
  * **keras.initializers.Initializer**
  * **keras.layers.Layer**  (for any layer, including activation functions)

* Like Custom Regularizer：不需要调用父类的`__call__`等方法。

```
class MyL1Regularizer(keras.regularizers.Regularizer):
    def __init__(self, factor):
        self.factor = factor
    def __call__(self, weights):
        return tf.reduce_sum(tf.abs(self.factor * weights))
    def get_config(self):
        return {"factor": self.factor}
```

因为`keras.regularizers.Regularizer`没有实现`__call__`和`get_config()`的：

```python
def __call__(self, x):
    """Compute a regularization penalty from an input tensor."""
    return 0.
```



* WARNING： 
  * must implement the `call()` method for **losses, layers (including activa‐ tion functions), and models,** 
  * or the`__call__() `method for **regularizers, initializers, and constraints.** 



#### Custom Metrics

