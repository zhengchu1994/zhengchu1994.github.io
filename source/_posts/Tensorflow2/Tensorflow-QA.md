---
title: Tensorflow2-QA
mathjax: true
date: 2020-10-01 21:53:32
tags: Tensorflow
categories: Tensorflow
visible:
---







**What is the difference between keras.evaluate() and keras.predict()?**

`model.predict`的结果是模型的输出`y_pred`，而`model.evaluate`返回根据`y_pred`设置的`metrics`。



The `model.evaluate` function predicts the output for the given input and then computes the metrics function specified in the` model.compile` and based on `y_true` and `y_pred` and returns the computed metric value as the output.

The `model.predict` just returns back the `y_pred`

Check here - the `predict_loop` used in `model.predict` [keras-team/keras](https://github.com/keras-team/keras/blob/bb9b800ebba8914b69db362cfac4e7c8a9b17a9e/keras/engine/training_arrays.py#L224)

and the `test_loop` used in `model.evaluate` [keras-team/keras](https://github.com/keras-team/keras/blob/bb9b800ebba8914b69db362cfac4e7c8a9b17a9e/keras/engine/training_arrays.py#L308)

They are same except the metrics computation part.

So if you use `model.predict` and then compute the metrics yourself, the computed metric value should turn out to be the same as `model.evaluate`

For example, one would use `model.predict` instead of `model.evaluate` in evaluating an *RNN/ LSTM based models* where the output needs to be fed as input in next time step as shown below.

