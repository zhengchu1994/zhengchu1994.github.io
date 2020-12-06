---
title: Tensorfow2复习--RNN
mathjax: true
date: 2020-07-04 19:00:00
tags: DeepLearning
categories: DeepLearning
visible:

---



# Tensorfow2.0--RNN



#### RNN cell

Input：Vector：$\mathbf{X}_{t-1}$，$\mathbf{y}_{t-1}$



a single layer, with a single neuron：

![a single layer, with a sin‐ gle neuron](https://tva1.sinaimg.cn/large/007S8ZIlly1ggerg9bbnuj30ww0ds76o.jpg)

```python
model = keras.models.Sequential([ keras.layers.SimpleRNN(1, input_shape=[None, 1])
])
```

By default, the SimpleRNN layer uses the **hyperbolic tangent activation function.** 

the initial state *h*(init) is set to 0

In a simple RNN, this output is also the new state *h*0. 

return one output per time step, you must set `return_sequences=True`



#### RNN layer

Input跟RNN cell一样，但是参数增加了，因为每个RNN cell对于$\mathbf{X}$还有$\mathbf{y}$分别有一个参数，多个RNN cell作为一层使得，有多个这样的参数。

![image-20200704111333418](https://tva1.sinaimg.cn/large/007S8ZIlly1ggerojiu9fj31a40fkadw.jpg)



#### Memory Cell

 New output , namely **hidden state**: $\mathbf{h_{(t)}}=f(\mathbf{x_{t}}, \mathbf{h_{(t-1)}})$



![image-20200704112055603](https://tva1.sinaimg.cn/large/007S8ZIlly1ggerw6bs28j310a0dadh3.jpg)





####  Deep RNNs

stack multiple layers of cells

![image-20200704112546983](https://tva1.sinaimg.cn/large/007S8ZIlly1gges1kszb4j31180g4gof.jpg)

```python
model = keras.models.Sequential([
keras.layers.SimpleRNN(20, return_sequences=True, input_shape=[None, 1]), keras.layers.SimpleRNN(20),
keras.layers.Dense(1)
])
```



#### Handling Long Sequences

Leading to : Unstable Gradients Problem

> nonsaturating activation functions (e.g., ReLU) may not help as much here; in fact, they may actually lead the RNN to be even more unstable during training
>
> Well, suppose Gradient Descent updates the weights in a way that increases the outputs slightly at the first time step. **Because the same weights are used at every time step, the outputs at the second time step may also be slightly increased**, and those at the third, and so on until the outputs explode—and a nonsaturating acti‐ vation function does not prevent that.



Solution: 

* *Layer Normalization* : it is very similar to Batch Normalization, but instead of normalizing across the batch dimension, it normalizes **across the features dimension.**

* Before activation



A cell must also have a `state_size` attribute and an `output_size` attribute.

```python
class LNSimpleRNNCell(keras.layers.Layer):
  def __init__(self, units, activation="tanh", **kwargs):
  super().__init__(**kwargs)
  self.state_size = units
  self.output_size = units
  self.simple_rnn_cell = keras.layers.SimpleRNNCell(units,
  activation=None) self.layer_norm = keras.layers.LayerNormalization()
  self.activation = keras.activations.get(activation) 
  def call(self, inputs, states):
   #new_states[0] equal to outputs
	outputs, new_states = self.simple_rnn_cell(inputs, states) 
  norm_outputs = self.activation(self.layer_norm(outputs)) return norm_outputs, [norm_outputs]
```

```python
model = keras.models.Sequential([ keras.layers.RNN(LNSimpleRNNCell(20),
                                                   return_sequences=True, 
                                                   input_shape=[None, 1]),
                                 keras.layers.RNN(LNSimpleRNNCell(20),
                                                  return_sequences=True),
                                 keras.layers.TimeDistributed(keras.layers.Dense(10)) ])

```



* all recurrent layers (except for keras.layers.RNN) and all cells provided by Keras have a **dropout** hyperparameter and a **recurrent_dropout** hyperparameter: the former defines the dropout rate to apply **to the inputs** (at each time step), and the latter defines the dropout rate for **the hidden states**



#### LSTM

* its state is split into two vectors: **h**(*t*)（short-term state） and **c**(*t*) （long-term state）

![image-20200704115106939](https://tva1.sinaimg.cn/large/007S8ZIlly1ggesrl0lkcj312w0mmdkc.jpg)

```python
model = keras.models.Sequential([
  keras.layers.LSTM(20, return_sequences=True, input_shape=[None, 1]),
  keras.layers.LSTM(20, return_sequences=True),
  keras.layers.TimeDistributed(keras.layers.Dense(10))
])
```



#### GRU cells

![image-20200704115624823](https://tva1.sinaimg.cn/large/007S8ZIlly1ggesx40uz3j31180os77h.jpg)



#### WaveNet

