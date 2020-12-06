---
title: Tensorflow2-tf.function
mathjax: true
date: 2020-10-01 21:53:32
tags: Tensorflow
categories: Tensorflow
visible:
---





`@tf.function`弥补Eager execution带来的效率问题：

- Debug in eager mode, then decorate with `@tf.function`.
- Don't rely on Python side effects like object mutation or list appends.
- [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) works best with TensorFlow ops; NumPy and Python calls are converted to constants.

1、当构建的计算图上只有少量特殊的ops时，时间效率差别不大:

```python
import timeit
conv_layer = tf.keras.layers.Conv2D(100, 3)

@tf.function
def conv_fn(image):
  return conv_layer(image)

image = tf.zeros([1, 200, 200, 100])
# warm up
conv_layer(image); conv_fn(image)
print("Eager conv:", timeit.timeit(lambda: conv_layer(image), number=10))
print("Function conv:", timeit.timeit(lambda: conv_fn(image), number=10))
print("Note how there's not much difference in performance for convolutions")
```



2、动态绑定：python具有动态绑定的语法特性，传递给函数不同类型的参数，函数有不同的行为，`tf.function`也可以做到，而且能够重用已有的计算图：

```python
@tf.function
def double(a):
  print("Tracing with", a)
  return a + a

print(double(tf.constant(1)))
print()
print(double(tf.constant(1.1)))
print()
print(double(tf.constant("a")))
print()


------
Tracing with Tensor("a:0", shape=(), dtype=int32)
tf.Tensor(2, shape=(), dtype=int32)

Tracing with Tensor("a:0", shape=(), dtype=float32)
tf.Tensor(2.2, shape=(), dtype=float32)

Tracing with Tensor("a:0", shape=(), dtype=string)
tf.Tensor(b'aa', shape=(), dtype=string)


---重用已有的graph
# This doesn't print 'Tracing with ...'
print(double(tf.constant("b")))
tf.Tensor(b'bb', shape=(), dtype=string)

```



可以用`print(double.pretty_printed_concrete_signatures())`查看已有的traces：

```python
double(a)
  Args:
    a: string Tensor, shape=()
  Returns:
    string Tensor, shape=()

double(a)
  Args:
    a: int32 Tensor, shape=()
  Returns:
    int32 Tensor, shape=()

double(a)
  Args:
    a: float32 Tensor, shape=()
  Returns:
    float32 Tensor, shape=()
```



对tensorflow的计算图的四点：

- A [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph) is the raw, language-agnostic, portable representation of your computation.
- A `ConcreteFunction` is an eagerly-executing wrapper around a [`tf.Graph`](https://www.tensorflow.org/api_docs/python/tf/Graph).
- A `Function` manages a cache of `ConcreteFunction`s and picks the right one for your inputs.
- [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function) wraps a Python function, returning a `Function` object



Every time a function is traced, a new concrete function is created. 

可以通过接口`get_concrete_function`得到concrete function

```python
print("Obtaining concrete trace")
double_strings = double.get_concrete_function(tf.constant("a"))
print("Executing traced function")
print(double_strings(tf.constant("a")))
print(double_strings(a=tf.constant("b")))

Obtaining concrete trace
Executing traced function
tf.Tensor(b'aa', shape=(), dtype=string)
tf.Tensor(b'bb', shape=(), dtype=string)


print(double_strings)
ConcreteFunction double(a)
  Args:
    a: string Tensor, shape=()
  Returns:
    string Tensor, shape=()
```



```python
# You can also call get_concrete_function on an InputSpec
double_strings_from_inputspec = double.get_concrete_function(tf.TensorSpec(shape=[], dtype=tf.string))
print(double_strings_from_inputspec(tf.constant("c")))


----
Tracing with Tensor("a:0", shape=(), dtype=string)
tf.Tensor(b'cc', shape=(), dtype=string)
```





Starting with TensorFlow 2.3, Python arguments remain in the signature, but are constrained to take the value set during tracing.

Tensorflow2.3开始，python传递的函数参数保留，并一直在tracing阶段保留：

```python
@tf.function
def pow(a, b):
  return a ** b

square = pow.get_concrete_function(a=tf.TensorSpec(None, tf.float32), b=2)
print(square)

----
ConcreteFunction pow(a, b=2)
  Args:
    a: float32 Tensor, shape=<unknown>
  Returns:
    float32 Tensor, shape=<unknown>
    
    
assert square(tf.constant(10.0)) == 100
#报错，因为b=2会被square一直保留
with assert_raises(TypeError):
  square(tf.constant(10.0), b=3)
```



When tracking down issues that only appear within [`tf.function`](https://www.tensorflow.org/api_docs/python/tf/function), here are some tips:

- Plain old Python `print` calls only execute during tracing, helping you track down when your function gets (re)traced.
- [`tf.print`](https://www.tensorflow.org/api_docs/python/tf/print) calls will execute every time, and can help you track down intermediate values during execution.
- [`tf.debugging.enable_check_numerics`](https://www.tensorflow.org/api_docs/python/tf/debugging/enable_check_numerics) is an easy way to track down where NaNs and Inf are created.
- `pdb` can help you understand what's going on during tracing. (Caveat: PDB will drop you into AutoGraph-transformed source code.)





