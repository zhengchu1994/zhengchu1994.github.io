---
title: Tensorflow2-CustomMetrics
mathjax: true
date: 2020-12-01 21:53:32
tags: Tensorflow
categories: Tensorflow
visible:
---











#### Keras未实现F1-score

以前Keras的metrics是错的，原因是这是非streaming的计算方式；



## Metrics

定义一个MSE作为metric如下，这是一个scalar常量值，并且在training或evaluation的时候，每个epoch看到的结果是该epoch下的每个batch的平均值；（is the average of the per-batch metric values for all batches see during a given epoch）

```python
def my_metric_fn(y_true, y_pred):
    squared_difference = tf.square(y_true - y_pred)
    return tf.reduce_mean(squared_difference, axis=-1)  # Note the `axis=-1`

model.compile(optimizer='adam', loss='mean_squared_error', metrics=[my_metric_fn])
```



但有时候我们想要的是在整个数据集上的metric，比如AUC等，所以上面的方法不适用，我们需要跨batches里，保持一个state，这个state记录我们最后要在总数据上计算的总metric；

做法是继承`Metric`类，它可以跨batches的保持一个state：

```python
class BinaryTruePositives(tf.keras.metrics.Metric):

  def __init__(self, name='binary_true_positives', **kwargs):
    super(BinaryTruePositives, self).__init__(name=name, **kwargs)
    self.true_positives = self.add_weight(name='tp', initializer='zeros')

  def update_state(self, y_true, y_pred, sample_weight=None):
    y_true = tf.cast(y_true, tf.bool)
    y_pred = tf.cast(y_pred, tf.bool)

    values = tf.logical_and(tf.equal(y_true, True), tf.equal(y_pred, True))
    values = tf.cast(values, self.dtype)
    if sample_weight is not None:
      sample_weight = tf.cast(sample_weight, self.dtype)
      values = tf.multiply(values, sample_weight)
    self.true_positives.assign_add(tf.reduce_sum(values))

  def result(self):
    return self.true_positives

  def reset_states(self):
    self.true_positives.assign(0)

m = BinaryTruePositives()
m.update_state([0, 1, 1, 1], [0, 1, 0, 0])
print('Intermediate result:', float(m.result()))

m.update_state([1, 1, 1, 1], [0, 1, 1, 0])
print('Final result:', float(m.result()))
```

* `add_metric()`接口如下：这样得到的是在每一个batch的平均值；（设置了`aggregation='mean'`）
* The quantity will then tracked under the name "activation_mean". The value tracked will be the average of the per-batch metric metric values (as specified by `aggregation='mean'`).

```python
class DenseLike(Layer):
  """y = w.x + b"""

  ...

  def call(self, inputs):
      output = tf.matmul(inputs, self.w) + self.b
      self.add_metric(tf.reduce_mean(output), aggregation='mean', name='activation_mean')
      return output
```







#### Streaming Metrics

我们需要流式的度量计算(*streaming metric* or *stateful metric*），一个batch计算后，下一个batch的metric计算时考虑上之前的计算值，从而得到增量的metric计算值。

创建streaming metric的方法是继承`keras.metrics.Metric class`：

* The constructor uses the `add_weight() `method to create **the variables needed to keep track** of the metric’s state over multiple batches；	**Keras tracks any tf.Variable that is set as an attribute**
* The `update_state() ` method is called when you use an instance of this class as a function;
*  When you use the metric as a function, the ` update_state() `method gets called first, then the` result()` method is called, and its output is returned.
* We also implement the `get_config()` method to ensure the threshold gets saved along with the model.
* When you define a metric using a simple function, Keras automatically calls it for each batch, and it keeps track of the mean during each epoch, just like we did manually. So the only benefit of our HuberMetric class is that the threshold will be saved. But of course, some metrics, **like precision, cannot simply be averaged over batches**: in those cases, there’s no other option than to implement a streaming metric.

```python
class HuberMetric(keras.metrics.Metric):
  def __init__(self, threshold=1.0, **kwargs):
    super().__init__(**kwargs) # handles base args (e.g., dtype) self.threshold = threshold
		self.huber_fn = create_huber(threshold)
		self.total = self.add_weight("total", initializer="zeros") 
    self.count = 	self.add_weight("count", initializer="zeros")
	def update_state(self, y_true, y_pred, sample_weight=None): 
    metric = self.huber_fn(y_true, y_pred) 
    self.total.assign_add(tf.reduce_sum(metric)) 
    self.count.assign_add(tf.cast(tf.size(y_true), tf.float32))
	def result(self):
    return self.total / self.count
	def get_config(self):
    base_config = super().get_config()
		return {**base_config, "threshold": self.threshold}
```







Anothe example：

```python
class CategoricalTruePositives(keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CategoricalTruePositives, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)


model = get_uncompiled_model()
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=[CategoricalTruePositives()],
)
model.fit(x_train, y_train, batch_size=64, epochs=3)
```













#### multiple inputs for metrics

```python
from keras.layers import *
from keras.models import *
import keras.backend as K


def get_divisor(x):
    return K.sqrt(K.sum(K.square(x), axis=-1))


def similarity(a, b):
    numerator = K.sum(a * b, axis=-1)
    denominator = get_divisor(a) * get_divisor(b)
    denominator = K.maximum(denominator, K.epsilon())
    return numerator / denominator


def max_margin_loss(positive, negative):
    #loss_matrix = K.maximum(0.0, 1.0 + negative - Reshape((1,))(positive))
    loss_matrix = K.maximum(0.0, 1.0 + negative - positive)
    loss = K.sum(loss_matrix, axis=-1, keepdims=True)
    return loss


def warp_loss(X):  
    z = X[0]
    positive_entity = X[1]
    negative_entities = X[2]
    positiveSim = similarity(z, positive_entity)
    #z_reshaped = Reshape((1, z.shape[1].value))(z)
    z_reshaped = K.expand_dims(z,axis=1)
    negativeSim = similarity(z_reshaped, negative_entities)
    #negativeSim = Reshape((negatives_titles.shape[1].value, 1,))
    negativeSim = K.expand_dims(negativeSim,axis=-1)
    loss = max_margin_loss(positiveSim, negativeSim)
    return loss


def warp_metricsX(X):
    z = X[0]
    positive_entity = X[1]
    negative_entities = X[2]
    positiveSim = similarity(z, positive_entity)
    #z_reshaped = Reshape((1, z.shape[1].value))(z)
    z_reshaped = K.expand_dims(z,axis=1)
    negativeSim = similarity(z_reshaped, negative_entities)
    #Reshape((negatives_titles.shape[1].value, 1,))
    negativeSim = K.expand_dims(negativeSim,axis=-1)

    position = K.sum(K.cast(K.greater(positiveSim, negativeSim), dtype="int32"), axis=1, keepdims=True)
    #accuracy = position / _NUMBER_OF_NEGATIVE_EXAMPLES
    accuracy = position / 30
    return accuracy


def mean_loss(yTrue,yPred):
    return K.mean(warp_loss(yPred))

def warp_metrics(yTrue,yPred):
    return warp_metricsX(yPred)


def build_nn_model():
    #wl, tl = load_vector_lookups()
    #embedded_layer_1 = initialize_embedding_matrix(wl)
    #embedded_layer_2 = initialize_embedding_matrix(tl)
    embedded_layer_1 =  Embedding(200,25)
    embedded_layer_2 =  Embedding(200,25)

    #sequence_input_1 = Input(shape=(_NUMBER_OF_LENGTH,), dtype='int32',name="text")
    sequence_input_1 = Input(shape=(30,), dtype='int32',name="text")
    sequence_input_positive = Input(shape=(1,), dtype='int32', name="positive")
    sequence_input_negatives = Input(shape=(10,), dtype='int32', name="negatives")

    embedded_sequences_1 = embedded_layer_1(sequence_input_1)
    #embedded_sequences_positive = Reshape((tl.shape[1],))(embedded_layer_2(sequence_input_positive))
    embedded_sequences_positive = Reshape((25,))(embedded_layer_2(sequence_input_positive))
    embedded_sequences_negatives = embedded_layer_2(sequence_input_negatives)

    conv_step1 = Convolution1D(
        filters=1000,
        kernel_size=5,
        activation="tanh",
        name="conv_layer_mp",
        padding="valid")(embedded_sequences_1)

    conv_step2 = GlobalMaxPooling1D(name="max_pool_mp")(conv_step1)
    conv_step3 = Activation("tanh")(conv_step2)
    conv_step4 = Dropout(0.2, name="dropout_mp")(conv_step3)
    #z = Dense(wl.shape[1], name="predicted_vec")(conv_step4) # activation="linear"
    z = Dense(25, name="predicted_vec")(conv_step4) # activation="linear"

    model = Model(
            inputs=[sequence_input_1, sequence_input_positive, sequence_input_negatives],
            outputs = [z,embedded_sequences_positive,embedded_sequences_negatives]
        )


    model.compile(loss=mean_loss, optimizer='adam',metrics=[warp_metrics])
    return model
```









