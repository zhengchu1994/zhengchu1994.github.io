---
title: Tensorflow2-Estimator
mathjax: true
date: 2020-10-01 21:53:32
tags: Tensorflow
categories: Tensorflow
visible:
---







Estimator封装了四个主要功能：

- training
- evaluation
- prediction
- export for serving



Esitmator提供了现在`tf.keras`正在构建中的功能：

- Parameter server based training
- Full [TFX](http://tensorflow.org/tfx) integration.



Pre-made Estimators封装了很多已有的networks，便于对不同模型结构做测试。



编写一个Estimator程序的步骤如下：



1. 数据处理接口：

   ```python
   def input_fn(dataset):
       ...  # manipulate dataset, extracting the feature dict and the label
       return feature_dict, label
   ```

2. 定义`feature columns`：通过接口`tf.feature_column`定义特征的name，types，预处理方式等。

   ```python
   # Define three numeric feature columns.
   population = tf.feature_column.numeric_column('population')
   crime_rate = tf.feature_column.numeric_column('crime_rate')
   median_education = tf.feature_column.numeric_column(
     'median_education',
     normalizer_fn=lambda x: x - global_education_mean) #包含数据处理
   ```

   

   3. 定义一个Estimator：

   ```python
   # Instantiate an estimator, passing the feature columns.
   estimator = tf.estimator.LinearClassifier(
     feature_columns=[population, crime_rate, median_education])
   ```

   

   4. training，evaluate等

   ```python
   # `input_fn` is the function created in Step 1
   estimator.train(input_fn=my_training_set, steps=2000)
   ```

   

   

   

   #### 从keras构建Estimator

   

   ```python
   keras_mobilenet_v2 = tf.keras.applications.MobileNetV2(
       input_shape=(160, 160, 3), include_top=False)
   keras_mobilenet_v2.trainable = False
   
   estimator_model = tf.keras.Sequential([
       keras_mobilenet_v2,
       tf.keras.layers.GlobalAveragePooling2D(),
       tf.keras.layers.Dense(1)
   ])
   
   # Compile the model
   estimator_model.compile(
       optimizer='adam',
       loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
       metrics=['accuracy'])
   
   # 此时使用的方法与任何其他pre-estimator一样
   est_mobilenet_v2 = tf.keras.estimator.model_to_estimator(keras_model=estimator_model)
   
   
   IMG_SIZE = 160  # All images will be resized to 160x160
   
   def preprocess(image, label):
     image = tf.cast(image, tf.float32)
     image = (image/127.5) - 1
     image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
     return image, label
   
   def train_input_fn(batch_size):
     data = tfds.load('cats_vs_dogs', as_supervised=True)
     train_data = data['train']
     train_data = train_data.map(preprocess).shuffle(500).batch(batch_size)
     return train_data
   
   est_mobilenet_v2.train(input_fn=lambda: train_input_fn(32), steps=500)
   
   
   ```

   

   

   

   ------

   #### feature_columns

   https://www.tensorflow.org/tutorials/structured_data/feature_columns

   

   特征列 通常用于对结构化数据实施特征工程时候使用，图像或者文本数据一般不会用到特征列。

   使用特征列可以将类别特征转换为one-hot编码特征，将连续特征构建分桶特征，以及对多个特征生成交叉特征等等。

   要创建特征列，请调用 tf.feature_column 模块的函数。该模块中常用的九个函数如下图所示，所有九个函数都会返回一个 Categorical-Column 或一个 Dense-Column 对象，但却不会返回 bucketized_column，后者继承自这两个类。

   注意：所有的Catogorical Column类型最终都要通过indicator_column转换成Dense Column类型才能传入模型！

   - numeric_column 数值列，最常用。

   - bucketized_column 分桶列，由数值列生成，可以由一个数值列出多个特征，one-hot编码。

   - categorical_column_with_identity 分类标识列，one-hot编码，相当于分桶列每个桶为1个整数的情况。

   - categorical_column_with_vocabulary_list 分类词汇列，one-hot编码，由list指定词典。

   - categorical_column_with_vocabulary_file 分类词汇列，由文件file指定词典。

   - categorical_column_with_hash_bucket 哈希列，整数或词典较大时采用。

   - indicator_column 指标列，由Categorical Column生成，one-hot编码

   - embedding_column 嵌入列，由Categorical Column生成，嵌入矢量分布参数需要学习。嵌入矢量维数建议取类别数量的 4 次方根。

   - crossed_column 交叉列，可以由除categorical_column_with_hash_bucket的任意分类列构成。

   

   ---------



#### What's the difference between a Tensorflow Keras Model and Estimator

[What's the difference between a Tensorflow Keras Model and Estimator?](https://stackoverflow.com/questions/51455863/whats-the-difference-between-a-tensorflow-keras-model-and-estimator)



#### 最大的差别就是分布式训练：

## Distribution

You can conduct distributed training across multiple servers with the Estimators API, but not with Keras API.

From the [Tensorflow Keras Guide](https://www.tensorflow.org/guide/keras), it says that:

> The Estimators API is used for training models for **distributed environments**.

And from the [Tensorflow Estimators Guide](https://www.tensorflow.org/guide/estimators#advantages_of_estimators), it says that:

> You can run Estimator-based models on a local host or on a **distributed multi-server** environment without changing your model. Furthermore, you can run Estimator-based models on CPUs, GPUs, or TPUs without recoding your model.




PS: Keras does handle low level operations, it's just not very standard. Its backend (`import keras.backend as K`) contains lots of functions that wrap around the backend functions. They're meant to be used in custom layers, custom metrics, custom loss functions, etc



#### Multi-worker training with Estimator

**Note:** While you can use Estimators with [`tf.distribute`](https://www.tensorflow.org/api_docs/python/tf/distribute) API, it's recommended to use Keras with [`tf.distribute`](https://www.tensorflow.org/api_docs/python/tf/distribute), see [multi-worker training with Keras](https://www.tensorflow.org/tutorials/distribute/multi_worker_with_keras). Estimator training with [`tf.distribute.Strategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) has limited support.（用keras更方便？）



when using Estimator for multi-worker training, it is necessary to shard the dataset by the number of workers to ensure model convergence. The input data is sharded by worker index, so that each worker processes `1/num_workers` distinct portions of the dataset.（需要把dataset均匀分布在多个worker上来保证收敛，根据worker的索引分配dataset）

Another reasonable approach to achieve convergence would be to shuffle the dataset with distinct seeds at each worker.(另一种保证收敛，对数据在每个worker上都进行shuffle操作。 )





```python
BUFFER_SIZE = 10000
BATCH_SIZE = 64

def input_fn(mode, input_context=None):
  datasets, info = tfds.load(name='mnist',
                                with_info=True,
                                as_supervised=True)
  mnist_dataset = (datasets['train'] if mode == tf.estimator.ModeKeys.TRAIN else
                   datasets['test'])

  def scale(image, label):
    image = tf.cast(image, tf.float32)
    image /= 255
    return image, label

  if input_context:
    mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                        input_context.input_pipeline_id)
  return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```



Write the layers, the optimizer, and the loss function for training. This tutorial defines the model with Keras layers, similar to the [multi-GPU training tutorial](https://www.tensorflow.org/tutorials/distribute/keras).

```python
LEARNING_RATE = 1e-4
def model_fn(features, labels, mode):
  model = tf.keras.Sequential([
      tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
      tf.keras.layers.MaxPooling2D(),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(64, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  logits = model(features, training=False)

  if mode == tf.estimator.ModeKeys.PREDICT:
    predictions = {'logits': logits}
    return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

  optimizer = tf.compat.v1.train.GradientDescentOptimizer(
      learning_rate=LEARNING_RATE)
  loss = tf.keras.losses.SparseCategoricalCrossentropy(
      from_logits=True, reduction=tf.keras.losses.Reduction.NONE)(labels, logits)
  loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
  if mode == tf.estimator.ModeKeys.EVAL:
    return tf.estimator.EstimatorSpec(mode, loss=loss)

  return tf.estimator.EstimatorSpec(
      mode=mode,
      loss=loss,
      train_op=optimizer.minimize(
          loss, tf.compat.v1.train.get_or_create_global_step()))
```

**Note:** Although the learning rate is fixed in this example, in general it may be necessary to adjust the learning rate based on the global batch size.(学习率根据全局batchsize动态变换更合适)





 It is also possible to distribute the evaluation via `eval_distribute`

```python
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()

config = tf.estimator.RunConfig(train_distribute=strategy)

classifier = tf.estimator.Estimator(
    model_fn=model_fn, model_dir='/tmp/multiworker', config=config)
tf.estimator.train_and_evaluate(
    classifier,
    train_spec=tf.estimator.TrainSpec(input_fn=input_fn),
    eval_spec=tf.estimator.EvalSpec(input_fn=input_fn)
)
```



