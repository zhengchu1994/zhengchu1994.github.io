---
title: Tensorflow2-Dataset
mathjax: true
date: 2020-10-01 21:53:32
tags: Tensorflow
categories: Tensorflow
visible:
---



#### tf.data

建立`tf.data.Dataset`代表数据：建立方法：`tf.data.Dataset.from_tensors()`或`tf.data.Dataset.from_tensor_slices()`，或`tf.data.TFRecordDataset()`.

只要建立了`Dataset`后，就可以像Scala一样进行数据变换生成新的数据；此外，`Dataset`是可迭代的：

```python
dataset = tf.data.Dataset.from_tensor_slices([8, 3, 0, 8, 2, 1])
for elem in dataset:
    print(elem.numpy())
---
8
3
0
8
2
1
```



```python
print(dataset.reduce(0, lambda state, value: state + value).numpy())
22
```



#### Dataset structure

`Dataset`里的每个元素类型为`tf.Tensor`,`tf.sparse.SparseTensor`,`tf.RaggedTensor`等。

`element_spec`属性可以告知元素的的类型信息：

```python
dataset2 = tf.data.Dataset.from_tensor_slices(
   (tf.random.uniform([4]),
    tf.random.uniform([4, 100], maxval=100, dtype=tf.int32)))

dataset2.element_spec

-----
(TensorSpec(shape=(), dtype=tf.float32, name=None),
 TensorSpec(shape=(100,), dtype=tf.int32, name=None))

# 自带了更多的方法
dataset1.element_spec.value_type
---
tensorflow.python.framework.ops.Tensor
```





#### 创建data

情况1:内存里可以放下，先转变为`tf.Tensor`，然后用`Dataset.from_tensor_slices()`处理为`Dataset`：

```python
train, test = tf.keras.datasets.fashion_mnist.load_data()
images, labels = train
images = images/255

dataset = tf.data.Dataset.from_tensor_slices((images, labels))
dataset
<TensorSliceDataset shapes: ((28, 28), ()), types: (tf.float64, tf.uint8)>
    
 # Note: The above code snippet will embed the features and labels arrays in your TensorFlow graph as tf.constant() operations.
```



TFRecord文件：是二进制的文件，可与`tf.data.TFRecordDataset`配合使用，形成pipeline：

```python
# Creates a dataset that reads all of the examples from two files.
fsns_test_file = tf.keras.utils.get_file("fsns.tfrec", "https://storage.googleapis.com/download.tensorflow.org/data/fsns-20160927/testdata/fsns-00000-of-00001")


dataset = tf.data.TFRecordDataset(filenames = [fsns_test_file])
raw_example = next(iter(dataset))
<TFRecordDatasetV2 shapes: (), types: tf.string>
    
#加载的数据还需要解码
parsed = tf.train.Example.FromString(raw_example.numpy())

```



文本数据：对应的接口`tf.data.TextLineDataset`:

```python
directory_url = 'https://storage.googleapis.com/download.tensorflow.org/data/illiad/'
file_names = ['cowper.txt', 'derby.txt', 'butler.txt']

# 这里开始下载数据；
file_paths = [
    tf.keras.utils.get_file(file_name, directory_url + file_name)
    for file_name in file_names
]

# 已被读入数据
dataset = tf.data.TextLineDataset(file_paths)
```



* 打乱这样的文本文件`Dataset.interleave`





对于CSV数据：

```python
titanic_file = tf.keras.utils.get_file("train.csv", "https://storage.googleapis.com/tf-datasets/titanic/train.csv")df = pd.read_csv(titanic_file, index_col=None)
df.head()
```





数据集分为`Batch`：

* 下面的例子用`dataset.batch(4)`将数据集分割为4个批量的大小；

```python
inc_dataset = tf.data.Dataset.range(100)
dec_dataset = tf.data.Dataset.range(0, -100, -1)
dataset = tf.data.Dataset.zip((inc_dataset, dec_dataset))
batched_dataset = dataset.batch(4)

for batch in batched_dataset.take(4):
  print([arr.numpy() for arr in batch])
```



但是`Dataset`的shape可能是`None`，原因是最后一个batch可能不足：

```python
batched_dataset
<BatchDataset shapes: ((None,), (None,)), types: (tf.int64, tf.int64)>
```

可以用`drop_remainder`忽略掉最后一个batch，这样所有的data都是一样的大小为`batch`；

```python
batched_dataset = dataset.batch(7, drop_remainder=True)
batched_dataset
```



处理的数据集里，如文本序列长度不一，可以用`padded_batch`接口，对长度不一的进行padding操作：

```python
dataset = tf.data.Dataset.range(100)
dataset = dataset.map(lambda x: tf.fill([tf.cast(x, tf.int32)], x))
for batch in dataset.take(10):
    print(batch.numpy())
    print()
    
-------
[]

[1]

[2 2]

[3 3 3]

[4 4 4 4]

[5 5 5 5 5]

[6 6 6 6 6 6]

[7 7 7 7 7 7 7]

[8 8 8 8 8 8 8 8]

[9 9 9 9 9 9 9 9 9]

# batch的大小为4，按照batch大小把dataset分组，每个batch都被padding为最长的元素长度
dataset = dataset.padded_batch(4, padded_shapes=(None, ))
for batch in dataset.take(10):
    print(batch.numpy())
    print()
    
------
[[0 0 0]
 [1 0 0]
 [2 2 0]
 [3 3 3]]

[[4 4 4 4 0 0 0]
 [5 5 5 5 5 0 0]
 [6 6 6 6 6 6 0]
 [7 7 7 7 7 7 7]]

[[ 8  8  8  8  8  8  8  8  0  0  0]
 [ 9  9  9  9  9  9  9  9  9  0  0]
 [10 10 10 10 10 10 10 10 10 10  0]
 [11 11 11 11 11 11 11 11 11 11 11]]

[[12 12 12 12 12 12 12 12 12 12 12 12  0  0  0]
 [13 13 13 13 13 13 13 13 13 13 13 13 13  0  0]
 [14 14 14 14 14 14 14 14 14 14 14 14 14 14  0]
 [15 15 15 15 15 15 15 15 15 15 15 15 15 15 15]]
.....
```



```python
dataset.repeat(3).batch(128) #这样的batch可能会跨越重复的dataset
dataset.batch(128).repeat(3) # 这样不会
```



打乱数据：`Dataset.shuffle()`；



#### 预处理data

`Dataset.map(f)`，其中`f`处理`tf.Tensor`型数据；

`tf.py_function()`：传python函数作为函数`f`；

```python
import scipy.ndimage as ndimage

def random_rotate_image(image):
  image = ndimage.rotate(image, np.random.uniform(-30, 30), reshape=False)
  return image


def tf_random_rotate_image(image, label):
  im_shape = image.shape
  [image,] = tf.py_function(random_rotate_image, [image], [tf.float32])
  image.set_shape(im_shape)
  return image, label

rot_ds = images_ds.map(tf_random_rotate_image)
```



加载数据集时，可以选择从训练集的某部分开始：

```python
range_ds = tf.data.Dataset.range(20)

iterator = iter(range_ds)
ckpt = tf.train.Checkpoint(step=tf.Variable(0), iterator=iterator)
manager = tf.train.CheckpointManager(ckpt, './', max_to_keep=3)

print([next(iterator).numpy() for _ in range(5)])

save_path = manager.save()

print([next(iterator).numpy() for _ in range(5)])

ckpt.restore(manager.latest_checkpoint)

print([next(iterator).numpy() for _ in range(5)])


---
[0, 1, 2, 3, 4]
[5, 6, 7, 8, 9]
# 又回来了第一次print之后的数据轮次
[5, 6, 7, 8, 9]
```





##  [`tf.data.TextLineDataset`](https://www.tensorflow.org/api_docs/python/tf/data/TextLineDataset)

