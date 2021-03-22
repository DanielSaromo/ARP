#@title Keras TF Implementation of Conv ARNN - By Daniel Saromo Mori and Matias Valdenegro Toro
# Adapted from Keras Implementation of Convolutional Layers:
# https://github.com/tensorflow/tensorflow/blob/v2.1.0/tensorflow/python/keras/layers/core.py#L1008-L1173
# Copyright 2015 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Conv ARNN implementation for Keras.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

import copy
import sys
import textwrap
import types as python_types
import warnings
 
import numpy as np

from tensorflow.keras.layers import Conv1D, Conv2D, Conv3D
from tensorflow.python.keras import activations

class AutoRotConv1D(Conv1D):
    def __init__(self,
               filters,
               kernel_size,
               xmin_lim,#n-dimensional vector
               xmax_lim,#n-dimensional vector
               L,
               xQ_scalar='auto',#if it is left in 'auto': xQ = 2*xmax-xmin
               eps = 1e-5,
               AutoRot=True,
               strides=1,
               padding='valid',
               data_format='channels_last',
               dilation_rate=1,
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):

        Conv1D.__init__(self, filters=filters, kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.xmin_lim = xmin_lim#
        self.xmax_lim = xmax_lim#

        if (xQ_scalar=='auto'):
            #Using information from the dataset and the activation function, calculates
            #the same xQ scalar value for all the neurons of the layer.
            self.xQ_scalar = 2*xmax_lim - xmin_lim
        else:
            self.xQ_scalar = xQ_scalar

        if ((L is None) and (AutoRot)):
            raise ValueError('L is None but the Auto-Rotation is turned on. With L = None, the Auto-Rotation flag must be set to False.')
        
        self.L = L
        self.AutoRot = AutoRot
        self.eps = eps

        self.arp_activation = activations.get(activation)

    def call(self, inputs):
        outputs = Conv1D.call(self, inputs)

        if self.AutoRot:
            inputs_xQ = self.xQ_scalar * tf.ones_like(inputs)
            inputs_xQ = tf.cast(inputs_xQ, self._compute_dtype)
            fOfxQ = Conv1D.call(self, inputs_xQ)
            
            #calculates rho
            den = tf.abs(fOfxQ)
            rho = tf.divide(self.L, den) + self.eps
            #applies rho multiplication
            outputs = tf.multiply(rho, outputs)

        #below, the outputs are activated
        if self.arp_activation is not None:
            return self.arp_activation(outputs)

        return outputs

    def get_config(self):
        config = {
            'xmin_lim': self.xmin_lim,
            'xmax_lim': self.xmax_lim,
            'xQ_scalar': self.xQ_scalar,
            'eps': self.eps,
            'AutoRot': self.AutoRot,
            'L': self.L,
            'activation': activations.serialize(self.arp_activation)
        }
        base_config = Conv1D.get_config(self)

        return dict(list(base_config.items()) + list(config.items()))

class AutoRotConv2D(Conv2D):
    def __init__(self,
               filters,
               kernel_size,
               xmin_lim,#n-dimensional vector
               xmax_lim,#n-dimensional vector
               L,
               xQ_scalar='auto',#if it is left in 'auto': xQ = 2*xmax-xmin
               eps = 1e-5,
               AutoRot=True,
               strides=(1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):

        Conv2D.__init__(self, filters=filters, kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.xmin_lim = xmin_lim#
        self.xmax_lim = xmax_lim#

        if (xQ_scalar=='auto'):
            #Using information from the dataset and the activation function, calculates
            #the same xQ scalar value for all the neurons of the layer.
            self.xQ_scalar = 2*xmax_lim - xmin_lim
        else:
            self.xQ_scalar = xQ_scalar

        if ((L is None) and (AutoRot)):
            raise ValueError('L is None but the Auto-Rotation is turned on. With L = None, the Auto-Rotation flag must be set to False.')

        self.L = L
        self.AutoRot = AutoRot
        self.eps = eps

        self.arp_activation = activations.get(activation)

    def call(self, inputs):
        outputs = Conv2D.call(self, inputs)

        if self.AutoRot:
            inputs_xQ = self.xQ_scalar * tf.ones_like(inputs)
            inputs_xQ = tf.cast(inputs_xQ, self._compute_dtype)
            fOfxQ = Conv2D.call(self, inputs_xQ)
            
            #calculates rho
            den = tf.abs(fOfxQ) + self.eps
            rho = tf.divide(self.L, den)
            #applies rho multiplication
            outputs = tf.multiply(rho, outputs)

        #below, the outputs are activated
        if self.arp_activation is not None:
            return self.arp_activation(outputs)

        return outputs

    def get_config(self):
        config = {
            'xmin_lim': self.xmin_lim,
            'xmax_lim': self.xmax_lim,
            'xQ_scalar': self.xQ_scalar,
            'eps': self.eps,
            'AutoRot': self.AutoRot,
            'L': self.L,
            'activation': activations.serialize(self.arp_activation)
        }
        base_config = Conv2D.get_config(self)

        return dict(list(base_config.items()) + list(config.items()))

class AutoRotConv3D(Conv3D):
    def __init__(self,
               filters,
               kernel_size,
               xmin_lim,#n-dimensional vector
               xmax_lim,#n-dimensional vector
               L,
               xQ_scalar='auto',#if it is left in 'auto': xQ = 2*xmax-xmin
               eps = 1e-5,
               AutoRot=True,
               strides=(1, 1, 1),
               padding='valid',
               data_format='channels_last',
               dilation_rate=(1, 1, 1),
               activation=None,
               use_bias=True,
               kernel_initializer='glorot_uniform',
               bias_initializer='zeros',
               kernel_regularizer=None,
               bias_regularizer=None,
               activity_regularizer=None,
               kernel_constraint=None,
               bias_constraint=None,
               **kwargs):

        Conv3D.__init__(self, filters=filters, kernel_size=kernel_size,
            strides=strides,
            padding=padding,
            data_format=data_format,
            dilation_rate=dilation_rate,
            activation=None,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            bias_constraint=bias_constraint,
            **kwargs)

        self.xmin_lim = xmin_lim#
        self.xmax_lim = xmax_lim#

        if (xQ_scalar=='auto'):
            #Using information from the dataset and the activation function, calculates
            #the same xQ scalar value for all the neurons of the layer.
            self.xQ_scalar = 2*xmax_lim - xmin_lim
        else:
            self.xQ_scalar = xQ_scalar

        if ((L is None) and (AutoRot)):
            raise ValueError('L is None but the Auto-Rotation is turned on. With L = None, the Auto-Rotation flag must be set to False.')

        self.L = L
        self.AutoRot = AutoRot
        self.eps = eps

        self.arp_activation = activations.get(activation)

    def call(self, inputs):
        outputs = Conv3D.call(self, inputs)

        if self.AutoRot:
            inputs_xQ = self.xQ_scalar * tf.ones_like(inputs)
            inputs_xQ = tf.cast(inputs_xQ, self._compute_dtype)
            fOfxQ = Conv3D.call(self, inputs_xQ)
            
            #calculates rho
            den = tf.abs(fOfxQ) + self.eps
            rho = tf.divide(self.L, den)
            #applies rho multiplication
            outputs = tf.multiply(rho, outputs)

        #below, the outputs are activated
        if self.arp_activation is not None:
            return self.arp_activation(outputs)

        return outputs

    def get_config(self):
        config = {
            'xmin_lim': self.xmin_lim,
            'xmax_lim': self.xmax_lim,
            'xQ_scalar': self.xQ_scalar,
            'eps': self.eps,
            'AutoRot': self.AutoRot,
            'L': self.L,
            'activation': activations.serialize(self.arp_activation)
        }
        base_config = Conv3D.get_config(self)

        return dict(list(base_config.items()) + list(config.items()))