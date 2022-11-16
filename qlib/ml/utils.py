#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov 15 12:41:02 2022

@author: archer
"""

from tensorflow.python.keras.engine.base_layer import Layer
from tensorflow.python.ops import array_ops, math_ops


from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

class AngleScaler(BaseEstimator, TransformerMixin):

    def __init__(self, freq=1, inverse=False):
        self.freq = freq
        self.inverse = inverse


    def fit(self, X, y=None):
        return self

    def transform(self, X):
        if self.inverse:
            self.inverse_transform(X)

        freq = self.freq
        return np.hstack([np.sin(freq*X),
                          np.cos(freq*X)])

    def inverse_transform(self, X):
        n = X.shape[1]//2
        return  np.arctan2(X[:, :n], X[:, n:]) / self.freq_inv


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
