---
title: Tensorflow2-DistributedTraining
mathjax: true
date: 2020-10-01 21:53:32
tags: Tensorflow
categories: Tensorflow
visible:
---









## 分布式训练







The code here is similar to the [multi-GPU training tutorial](https://www.tensorflow.org/tutorials/distribute/keras) with one key difference: 

when using Estimator for multi-worker training, it is necessary to shard the dataset by the number of workers to ensure model convergence. （ multi-worker 模式下的分布式模式下，作为包证模型收敛的手段，数据集切割分配到多个worker上。）

The input data is sharded by worker index, so that each worker processes `1/num_workers` distinct portions of the dataset.（分布式中的workers用以分摊训练集，输入的数据根据给定的id分配到worker上，id指定了分配的worker。）

```python
import tensorflow_datasets as tfds
import tensorflow as tf
tfds.disable_progress_bar()

import os, json


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

  # 若存在分布式的上下文，则对数据进行分配
  if input_context:
    mnist_dataset = mnist_dataset.shard(input_context.num_input_pipelines,
                                        input_context.input_pipeline_id)
  return mnist_dataset.map(scale).cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
```









# Multi-worker training with Keras



multi-worker distributed training with Keras model using [`tf.distribute.Strategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) API, specifically [`tf.distribute.experimental.MultiWorkerMirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy).（strategy是多workers模式的接口，使得单机keras模型的代码可以无缝迁移到多workers环境。）



准备工作：

```python
import os
import tensorflow as tf
import numpy as np

def mnist_dataset(batch_size):
  (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
  # The `x` arrays are in uint8 and have values in the range [0, 255].
  # We need to convert them to float32 with values in the range [0, 1]
  x_train = x_train / np.float32(255)
  y_train = y_train.astype(np.int64)
  train_dataset = tf.data.Dataset.from_tensor_slices(
      (x_train, y_train)).shuffle(60000).repeat().batch(batch_size)
  return train_dataset

def build_and_compile_cnn_model():
  model = tf.keras.Sequential([
      tf.keras.Input(shape=(28, 28)),
      tf.keras.layers.Reshape(target_shape=(28, 28, 1)),
      tf.keras.layers.Conv2D(32, 3, activation='relu'),
      tf.keras.layers.Flatten(),
      tf.keras.layers.Dense(128, activation='relu'),
      tf.keras.layers.Dense(10)
  ])
  model.compile(
      loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
      optimizer=tf.keras.optimizers.SGD(learning_rate=0.001),
      metrics=['accuracy'])
  return model


per_worker_batch_size = 64
single_worker_dataset = mnist_dataset(per_worker_batch_size)
single_worker_model = build_and_compile_cnn_model()
single_worker_model.fit(single_worker_dataset, epochs=3, steps_per_epoch=70)

```







TensorFlow里, `TF_CONFIG`是设置多机器训练环境变量的JSON串, 每一个机器很可能有不同的任务. 



`TF_CONFIG`由两部分组成： `cluster` 和 `task`. 



`cluster` provides information about the training cluster, which is a dict consisting of different types of jobs such as `worker`. 



In multi-worker training with `MultiWorkerMirroredStrategy`, there is usually one `worker` that takes on a little more responsibility like **saving checkpoint and writing summary file** for TensorBoard in addition to what a regular `worker` does. (Chief worker执行保存checkpoint和summary的任务)



Such worker is referred to as the `chief` worker, and it is customary that the `worker` with `index` 0 is appointed as the chief `worker` (in fact this is how `tf.distribute.Strategy` is implemented). （chief worker 的index为0）



`task` on the other hand provides information of the current task. （task提供当前任务的相关信息）



The first component `cluster` is the same for all workers, and the second component `task` is different on each worker and specifies the `type` and `index` of that worker. （cluster对于所有的workers都一样，task对不同的worker有不同的安排。）



In this example, we set the task `type` to `"worker"` and the task `index` to `0`. This means the machine that has such setting is the first worker, which will be appointed as the chief worker and do more work than other workers. （分配的task信息里标记有index=0，type=worker的机器被认为是chief。）



Note that other machines will need to have `TF_CONFIG` environment variable set as well, and it should have the same `cluster` dict, but different task `type` or task `index` depending on what the roles of those machines are.

（其他机器的角色也根据获得的task信息而做配置。）



For illustration purposes, this tutorial shows how one may set a `TF_CONFIG` with 2 workers on `localhost`.  In practice, users would create multiple workers on external IP addresses/ports, and set `TF_CONFIG` on each worker appropriately.

```python
# 配置两个workers 在localhost上。
os.environ['TF_CONFIG'] = json.dumps({
    'cluster': {
        'worker': ["localhost:12345", "localhost:23456"]
    },
    'task': {'type': 'worker', 'index': 0}
})
```





## Choose the right strategy



tensorflow上的分布式训练分为两种，同步与异步训练两种方式；同步训练时，所有的变量都被复制到每一个workers上，



In TensorFlow, distributed training consists of synchronous training, where the steps of training are synced across the workers and replicas, and asynchronous training, where the training steps are not strictly synced.（同步与异步训练两种方式）



`MultiWorkerMirroredStrategy`, which is the recommended strategy for **synchronous** multi-worker training, will be demonstrated in this guide. To train the model, use an instance of [`tf.distribute.experimental.MultiWorkerMirroredStrategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/experimental/MultiWorkerMirroredStrategy).（该接口执行同步训练） `MultiWorkerMirroredStrategy` creates copies of all variables in the model's layers on each device across all workers.（所有的变量都被复制到每一个workers上。）

 It uses `CollectiveOps`, a TensorFlow op for collective communication, to aggregate gradients and keep the variables in sync. The [`tf.distribute.Strategy` guide](https://www.tensorflow.org/guide/distributed_training) has more details about this strategy.（它通过`CollectiveOps`操作使得变量间的梯度保持同步。）



```python
strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy()
```



**Note:** `TF_CONFIG` is parsed and TensorFlow's GRPC servers are started at the time `MultiWorkerMirroredStrategy()` is called, so `TF_CONFIG` environment variable must be set before a [`tf.distribute.Strategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) instance is created.（在`strategy`策略执行前，必须初始化`TF_CONFIG`。)



`MultiWorkerMirroredStrategy` provides multiple implementations via the [`CollectiveCommunication`](https://github.com/tensorflow/tensorflow/blob/a385a286a930601211d78530734368ccb415bee4/tensorflow/python/distribute/cross_device_ops.py#L928) parameter. `RING` implements ring-based collectives using gRPC as the cross-host communication layer. `NCCL` uses [Nvidia's NCCL](https://developer.nvidia.com/nccl) to implement collectives. `AUTO` defers the choice to the runtime. The best choice of collective implementation depends upon the number and kind of GPUs, and the network interconnect in the cluster。（`MultiWorkerMirroredStrategy`提供了多种实现方式，可以通过参数`CollectiveCommunication`设置）



##### Train the model with MultiWorkerMirroredStrategy



With the integration of [`tf.distribute.Strategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) API into [`tf.keras`](https://www.tensorflow.org/api_docs/python/tf/keras), the only change you will make to distribute the training to multi-worker is enclosing the model building and `model.compile()` call inside `strategy.scope()`. The distribution strategy's scope dictates how and where the variables are created, and in the case of `MultiWorkerMirroredStrategy`, the variables created are `MirroredVariable`s, and they are replicated on each of the workers.（`strategy.scope()`告知了变量在哪里被什么模型所创建。）





**Note:** Currently there is a limitation in `MultiWorkerMirroredStrategy` where TensorFlow ops need to be created after the instance of strategy is created. If you see `RuntimeError: Collective ops must be configured at program startup`, try creating the instance of `MultiWorkerMirroredStrategy` at the beginning of the program and put the code that may create ops after the strategy is instantiated.

（`RuntimeError: Collective ops must be configured at program startup` 这个bug怎么处理）





**Note:** If you have an infinite dataset (by calling `.repeat()` on the dataset), you must specify the number of steps to run through `steps_per_epoch` argument to `model.fit()`. 

In that case, `model.fit()` does not create a new iterator from the input every epoch, but continues from wherever the last epoch ended.

 If you have a finite dataset, setting `steps_per_epoch` is optional. In particular, if the sharding is not balanced (for example, this could happen if you have a file-based dataset with the number of files more than the number of workers and some workers get files that contain more data than others. You can shard the data more evenly by manually setting [`tf.data.experimental.AutoShardPolicy`](https://www.tensorflow.org/api_docs/python/tf/data/experimental/AutoShardPolicy), more details [here](https://www.tensorflow.org/tutorials/distribute/input#sharding)), and `steps_per_epoch` is not set or set to be greater than the size of the smallest shard divided by the per-worker batch size, you might get partial batches towards the end of training. （数据的需求不同，在多个worker上怎么平衡。）

```python
num_workers = 4

# Here the batch size scales up by number of workers since 
# `tf.data.Dataset.batch` expects the global batch size. Previously we used 64, 
# and now this becomes 128.
global_batch_size = per_worker_batch_size * num_workers
multi_worker_dataset = mnist_dataset(global_batch_size)

with strategy.scope():
  # Model building/compiling need to be within `strategy.scope()`.
  multi_worker_model = build_and_compile_cnn_model()

# Keras' `model.fit()` trains the model with specified number of epochs and
# number of steps per epoch. Note that the numbers here are for demonstration
# purposes only and may not sufficiently produce a model with good quality.
multi_worker_model.fit(multi_worker_dataset, epochs=3, steps_per_epoch=70)
```





`fit`接口里`steps_per_epoch`的意思：一个`epoch`传递多少次数据，每次传递的数据大小都为batch-size。

`steps_per_epoch`：Since Keras data generator is meant to loop infinitely, `steps_per_epoch` indicates how many times you will fetch a new batch from generator during single epoch. Therefore, if you simply take `steps_per_epoch = int(number_of_train_samples / batch_size)`, your last batch would have less than `batch_size` items and would be discarded（每一个epoch的数据量 = steps_per_epoch * batch_size）

`steps_per_epoch` is the number of batches of your set batch size is ran through the network in one epoch.

You have set your `steps_per_epoch` to be `training_set_size//batch_size` for a good reason. This ensures all data are trained upon in one epoch, providing the number divides exactly (if not it rounds by the // operator).



##### Dataset sharding and batch size

In multi-worker training with `MultiWorkerMirroredStrategy`, sharding the dataset is needed to ensure convergence and performance. However, note that in above code snippet, the datasets are directly passed to `model.fit()` without needing to shard; this is because [`tf.distribute.Strategy`](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) API takes care of the dataset sharding automatically. It shards the dataset at the file level which may create skewed shards. In extreme cases where there is only one file, only the first shard (i.e. worker) will get training or evaluation data and as a result all workers will get errors. （该API已经实现了分布式下合理的数据分割）

If you prefer manual sharding for your training, automatic sharding can be turned off via [`tf.data.experimental.DistributeOptions`](https://www.tensorflow.org/api_docs/python/tf/data/experimental/DistributeOptions) api. Concretely,



----------------



~~~python
# parse the TF_CONFIG
resolver = tf.distribute.cluster_resolver.TFConfigClusterResolver()
cluster_spec = resolver.cluster_spec().as_dict()

----------
class TFConfigClusterResolver(ClusterResolver):
  """Implementation of a ClusterResolver which reads the TF_CONFIG EnvVar.

  This is an implementation of cluster resolvers when using TF_CONFIG to set
  information about the cluster. The cluster spec returned will be
  initialized from the TF_CONFIG environment variable.

  An example to set TF_CONFIG is:

    ```Python
    os.environ['TF_CONFIG'] = json.dumps({
      'cluster': {
          'worker': ["localhost:12345", "localhost:23456"]
      },
      'task': {'type': 'worker', 'index': 0}
    })
    ```

  However, sometimes the container orchestration framework will set TF_CONFIG
  for you. In this case, you can just create an instance without passing in any
  arguments. You can find an example here to let Kuburnetes set TF_CONFIG for
  you: https://github.com/tensorflow/ecosystem/tree/master/kubernetes. Then you
  can use it with `tf.distribute.Strategy` as:

    ```Python
    # `TFConfigClusterResolver` is already the default one in the following
    # strategy.
    strategy = tf.distribute.experimental.MultiWorkerMirroredStrategy(
        cluster_resolver=TFConfigClusterResolver())
    ```
  """

  def __init__(self,
               task_type=None,
               task_id=None,
               rpc_layer=None,
               environment=None):
    """Creates a new TFConfigClusterResolver.

    Args:
      task_type: (String, optional) Overrides the task type specified in the
        TF_CONFIG environment variable.
      task_id: (Integer, optional) Overrides the task index specified in the
        TF_CONFIG environment variable.
      rpc_layer: (String, optional) Overrides the rpc layer TensorFlow uses.
      environment: (String, optional) Overrides the environment TensorFlow
        operates in.
    """



---------
def cluster_spec(self):
  """Returns a ClusterSpec based on the TF_CONFIG environment variable.

      Returns:
        A ClusterSpec with information from the TF_CONFIG environment variable.
      """
  tf_config = _load_tf_config()
  if 'cluster' not in tf_config:
    return ClusterSpec({})
  return ClusterSpec(tf_config['cluster'])

------

config_pb2.RunOptions

_warm_start_settings
~~~

