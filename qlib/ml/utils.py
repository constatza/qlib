#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:41:02 2022

@author: archer
"""

import tensorflow as tf
import numpy as np

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops, math_ops



class SinCosTransformation(Layer):
    """Multiply inputs by `scale` and adds `offset`.
    For instance:
    1. To rescale an input in the `[0, 255]` range
    to be in the `[0, 1]` range, you would pass `scale=1./255`.
    2. To rescale an input in the `[0, 255]` range to be in the `[-1, 1]` range,
    you would pass `scale=1./127.5, offset=-1`.
    The rescaling is applied both during training and inference.
    Input shape:
      Arbitrary.
    Output shape:
      Same as input.
    Arguments:
      scale: Float, the scale to apply to the inputs.
      offset: Float, the offset to apply to the inputs.
      name: A string, the name of the layer.
    """

    def __init__(self, scale=1, inverse=False, name=None, **kwargs):
        self.scale = scale
        self.inverse=inverse

        super(SinCosTransformation, self).__init__(name=name, **kwargs)

    def call(self, inputs):
        if self.inverse:
            dtype = self._compute_dtype
            scale = math_ops.cast(self.scale, dtype)
            n = array_ops.shape(inputs)[1]//2
            siny = math_ops.cast(inputs[:, :n], dtype)
            cosy = math_ops.cast(inputs[:, n:], dtype)

            return math_ops.atan2(siny, cosy) / scale
        else:
            dtype = self._compute_dtype
            scale = math_ops.cast(self.scale, dtype)

            y = math_ops.cast(inputs, dtype) * scale
            return  array_ops.concat([math_ops.sin(y), math_ops.cos(y)], 1)



    def compute_output_shape(self, input_shape):
        if self.inverse:
            return input_shape//2
        else:
            return 2*input_shape

    def get_config(self):
        config = {
        'scale': self.scale,
        'inverse': self.inverse
        }
        base_config = super(SinCosTransformation, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))


class KNNRegressor(tf.keras.Model):

    def __init__(self, k, dtype_policy=tf.keras.mixed_precision.Policy('float32')):
        super(KNNRegressor, self).__init__()
        self.k = k
        self._dtype_policy = dtype_policy
        
    def fit(self, X_train, y_train):
        self.y_train = tf.cast(y_train, tf.float32)
        self.X_train = self.add_weight(shape=X_train.shape, initializer='uniform', trainable=False)
        self.X_train.assign(tf.cast(X_train, tf.float32))
        return self
        
    def __call__(self, inputs):
        expanded_a = tf.expand_dims(tf.cast(inputs, tf.float32), 1)
        expanded_b = tf.expand_dims(self.X_train, 0)

        # Calculate distance between input and stored weights
        distances = tf.reduce_sum(tf.math.squared_difference(expanded_a, expanded_b), 2)
        
        # Find nearest k neighbors
        _, indices = tf.nn.top_k(-distances, k=self.k)
        nearest_neighbors = tf.gather(self.y_train, indices)
        
        # Calculate mean of nearest neighbors as output
        output = tf.reduce_mean(nearest_neighbors, axis=1)
        return output
