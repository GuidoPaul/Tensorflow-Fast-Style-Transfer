#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Copyright (c) 2015-2016 Anish Athalye. Released under GPLv3.

import tensorflow as tf
import numpy as np
import scipy.io

# download URL : http://www.vlfeat.org/matconvnet/models/imagenet-vgg-verydeep-19.mat
MODEL_FILE_NAME = 'imagenet-vgg-verydeep-19.mat'


class VGG19(object):
    layers = ('conv1_1', 'relu1_1', 'conv1_2', 'relu1_2', 'pool1', 'conv2_1',
              'relu2_1', 'conv2_2', 'relu2_2', 'pool2', 'conv3_1', 'relu3_1',
              'conv3_2', 'relu3_2', 'conv3_3', 'relu3_3', 'conv3_4', 'relu3_4',
              'pool3', 'conv4_1', 'relu4_1', 'conv4_2', 'relu4_2', 'conv4_3',
              'relu4_3', 'conv4_4', 'relu4_4', 'pool4', 'conv5_1', 'relu5_1',
              'conv5_2', 'relu5_2', 'conv5_3', 'relu5_3', 'conv5_4', 'relu5_4')

    def __init__(self, data_path):
        data = scipy.io.loadmat(data_path)

        self.mean_pixel = np.array([123.68, 116.779, 103.939])

        # mean = data['normalization'][0][0][0]
        # mean_pixel = np.mean(mean, axis=(0, 1))
        self.weights = data['layers'][0]

    def _conv_layer(self, input, weights, bias):
        conv = tf.nn.conv2d(
            input, tf.constant(weights), strides=(1, 1, 1, 1), padding='SAME')
        return tf.nn.bias_add(conv, bias)

    def _pool_layer(self, input):
        return tf.nn.max_pool(
            input, ksize=(1, 2, 2, 1), strides=(1, 2, 2, 1), padding='SAME')

    def preprocess(self, image):
        return image - self.mean_pixel

    def unprocess(self, image):
        return image + self.mean_pixel

    def feed_forward(self, input_image, scope=None):
        net = {}
        current = input_image

        with tf.variable_scope(scope):
            for i, name in enumerate(self.layers):
                kind = name[:4]
                if kind == 'conv':
                    kernels, bias = self.weights[i][0][0][2][0]
                    # matconvnet: weights are [width, height, in_channels, out_channels]
                    # tensorflow: weights are [height, width, in_channels, out_channels]
                    kernels = np.transpose(kernels, (1, 0, 2, 3))
                    bias = bias.reshape(-1)
                    current = self._conv_layer(current, kernels, bias)
                elif kind == 'relu':
                    current = tf.nn.relu(current)
                elif kind == 'pool':
                    current = self._pool_layer(current)
                net[name] = current

        assert len(net) == len(self.layers)
        return net
