---
title: Tensorfow2复习--初步Keras
mathjax: true
date: 2020-06-12 19:00:00
tags: DeepLearning
categories: DeepLearning
visible:

---



# Tensorfow2.0--初步Keras



### 激活函数

1、Sigmoid函数，又称为Logistic函数，特点：输出范围[0，1]。

```python
def Sigmoid(z):
  return 1 / (1 + np.exp(-z))
```



2、Tanh函数：$\text{Tanh}(z)=2*\text{Sigmoid}(2z) - 1$ ，特点：output范围在[-1， 1]之间，效果是在训练初期每层网络的输出都或多或少的聚集在0附近，**可以加快收敛**。



3、ReLU函数，特点：0点不可导。



```python
def relu(z):
    return np.maximum(0, z)
```



可以用**两点估计法**计算下函数梯度：

```python
def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)
```





### MLP



```python
model = keras.models.Sequential()
#没有参数，只是对输入数据做变换: receives input data X, it computes X.reshape(-1, 1).
model.add(keras.layers.Flatten(input_shape=[28, 28]))
model.add(keras.layers.Dense(300, activation="relu"))
model.add(keras.layers.Dense(100, activation="relu"))
model.add(keras.layers.Dense(10, activation="softmax"))


#简化形式
model = keras.models.Sequential([
    keras.layers.Flatten(input_shape=[28, 28]),
    keras.layers.Dense(300, activation="relu"),
    keras.layers.Dense(100, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])
```



```python
## tf.keras.backend.clear_session will discard the values resides in the variable defined in the graph
keras.backend.clear_session()
np.random.seed(42)
tf.random.set_seed(42)
```



可用得到模型相关信息：

```
keras.utils.plot_model(model, "my_fashion_mnist_model.png", show_shapes=True)
model.summary()
model.layers
```



设置损失，优化器，度量函数：

`sparse_categorical_cross entropy`：类别里存一个数的指示（**Sparse labels**），比如3表示第三类。

`categorical_crossentropy`：相反是**one-hot labels**，如 [ 0,  0,  1 ] 来表示属于第三类。

转换：`keras.utils.to_categorical() function`用于转换sparse labels to one-hot labels

```python
model.compile(loss="sparse_categorical_crossentropy",
              optimizer="sgd",
              metrics=["accuracy"])
# 全名
model.compile(loss=keras.losses.sparse_categorical_crossentropy,
              optimizer=keras.optimizers.SGD(),
              metrics=[keras.metrics.sparse_categorical_accuracy])
```



设置验证集：`validation_split=0.1` tells Keras to use the last 10% of the data (before shuffling) for validation.

* **Tips1**:  **overfitting the training set** ：训练集上的效果很大程度上优于验证集。



字典信息：

```python
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid))
                    
pd.DataFrame(history.history).plot(figsize=(8, 5)) plt.grid(True)
plt.gca().set_ylim(0, 1) # set the vertical range to [0-1] plt.show()                
```





* 训练集是每轮里跑的均值： **the validation error is computed at the *end* of each epoch, while the training error is computed using a running mean *during* each epoch**。



预测：

```python
y_proba = model.predict(X_new)
y_pred = model.predict_classes(X_new)
```





###  Wide & Deep neural network

![image-20200614094252714](https://tva1.sinaimg.cn/large/007S8ZIlly1gfrko05d25j313y0jsgoe.jpg)

```python
input_ = keras.layers.Input(shape=X_train.shape[1:])
hidden1 = keras.layers.Dense(30, activation="relu")(input_) # 函数式传递不错
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_, hidden2])
output = keras.layers.Dense(1)(concat)
model = keras.models.Model(inputs=[input_], outputs=[output])
```





* 增加多个输入，多个输出的网络结构：

  

  ![image-20200614100256233](https://tva1.sinaimg.cn/large/007S8ZIlly1gfrl8vgo6vj31560iijty.jpg)

```python
input_A = keras.layers.Input(shape=[5], name="wide_input") #
input_B = keras.layers.Input(shape=[6], name="deep_input") #
hidden1 = keras.layers.Dense(30, activation="relu")(input_B)
hidden2 = keras.layers.Dense(30, activation="relu")(hidden1)
concat = keras.layers.concatenate([input_A, hidden2])
output = keras.layers.Dense(1, name="main_output")(concat) #
aux_output = keras.layers.Dense(1, name="aux_output")(hidden2) #
model = keras.models.Model(inputs=[input_A, input_B],
                           outputs=[output, aux_output])
```



对两部分可以分配不同的损失函数，权重：

```python
model.compile(loss=["mse", "mse"], loss_weights=[0.9, 0.1],
              optimizer=keras.optimizers.SGD(lr=1e-3))

history = model.fit([X_train_A, X_train_B], [y_train, y_train], epochs=20,
                    validation_data=([X_valid_A, X_valid_B], [y_valid, y_valid]))
```





通过继承实现同样的模型，

优势：支持更复杂的操作，如条件，循环等；缺点： Keras cannot save or clone it；when you call the **summary()** method, you only get a list of layers, without any information on how they are connected to each other.

```python
class WideAndDeepModel(keras.models.Model):
    def __init__(self, units=30, activation="relu", **kwargs):
        super().__init__(**kwargs)
        self.hidden1 = keras.layers.Dense(units, activation=activation)
        self.hidden2 = keras.layers.Dense(units, activation=activation)
        self.main_output = keras.layers.Dense(1)
        self.aux_output = keras.layers.Dense(1)
        
    def call(self, inputs):
        input_A, input_B = inputs
        hidden1 = self.hidden1(input_B)
        hidden2 = self.hidden2(hidden1)
        concat = keras.layers.concatenate([input_A, hidden2])
        main_output = self.main_output(concat)
        aux_output = self.aux_output(hidden2)
        return main_output, aux_output

model = WideAndDeepModel(30, activation="relu")
```





### 存储和加载模型

```python
model.save("my_keras_model.h5")
model = keras.models.load_model("my_keras_model.h5")

model.save_weights("my_keras_weights.ckpt")
model.load_weights("my_keras_weights.ckpt")
```

#### 训练过程中对checkpoint做保存

```python
# save_best_only=True，
checkpoint_cb = keras.callbacks.ModelCheckpoint("my_keras_model.h5", save_best_only=True)
model = keras.models.load_model("my_keras_model.h5") # rollback to best model
mse_test = model.evaluate(X_test, y_test)


# 早停，继续最好的权重
early_stopping_cb = keras.callbacks.EarlyStopping(patience=10,
                                                  restore_best_weights=True)

history = model.fit(X_train, y_train, epochs=100,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, early_stopping_cb])
mse_test = model.evaluate(X_test, y_test)
```





### Tensorboard

```python
root_logdir = os.path.join(os.curdir, "my_logs")

def get_run_logdir():
    import time
    run_id = time.strftime("run_%Y_%m_%d-%H_%M_%S")
    return os.path.join(root_logdir, run_id)

# keras自带传递一个生产保存文件地址的函数
tensorboard_cb = keras.callbacks.TensorBoard(run_logdir)
history = model.fit(X_train, y_train, epochs=30,
                    validation_data=(X_valid, y_valid),
                    callbacks=[checkpoint_cb, tensorboard_cb])
```





```bash
$ tensorboard --logdir=./my_logs --port=6006
```



在jupyter上直接看：

```python
%load_ext tensorboard
%tensorboard --logdir=./my_logs --port=6006
```







### Fine-Tuning

* 1、**Grid—Search**：对超参数定义可选的list，然后组合；
  * 一般GridSearchCV的参数`refit=True`，即选到了某个超参的最优值后，固定这个超参数为它，然后去塞选其他超参数的值。

```python
from sklearn.model_selection import GridSearchCV
param_grid = [
  {'n_estimatoor': [3, 10, 30], 'max_features': [2, 4, 6, 8]}, #12种组合
  {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2 ,3, 4]}, # 6种组合
]
```

* 2、**Randomized Search**：selecting  a random value for each hyperparameter at every iterationn.
* 3、**Ensemble Methods**：把所有好的模型组合起来做分析。



-----



#### Keras + GridSearch：

1、Any extra parameter you pass to the `fit()` method will get passed to the underlying Keras model. 

```python
def build_model(n_hidden=1, n_neurons=30, learning_rate=3e-3, input_shape=[8]):
    model = keras.models.Sequential()
    model.add(keras.layers.InputLayer(input_shape=input_shape))
    for layer in range(n_hidden):
        model.add(keras.layers.Dense(n_neurons, activation="relu"))
    model.add(keras.layers.Dense(1))
    optimizer = keras.optimizers.SGD(lr=learning_rate)
    model.compile(loss="mse", optimizer=optimizer)
    return model
  
 keras_reg = keras.wrappers.scikit_learn.KerasRegressor(build_model)

# 额外的参数被传给keras
keras_reg.fit(X_train, y_train, epochs=100,
              validation_data=(X_valid, y_valid),
              callbacks=[keras.callbacks.EarlyStopping(patience=10)])

```



#### Keras + RandomizedSearch：

```python
from scipy.stats import reciprocal
from sklearn.model_selection import RandomizedSearchCV

param_distribs = {
    "n_hidden": [0, 1, 2, 3],
    "n_neurons": np.arange(1, 100),
    "learning_rate": reciprocal(3e-4, 3e-2),
}

rnd_search_cv = RandomizedSearchCV(keras_reg, param_distribs, n_iter=10, cv=3, verbose=2)
rnd_search_cv.fit(X_train, y_train, epochs=100,
                  validation_data=(X_valid, y_valid),
                  callbacks=[keras.callbacks.EarlyStopping(patience=10)])
```







