from __future__ import absolute_import, print_function

import numpy as np
import tensorflow as tf

from utils import split_and_gate


############################################################
##  refactored module to have homogenous and simpler api  ##
############################################################


def gated_cnn_layer(X, state, kernel_shape, name):
    """
    Gated PixelCNN layer for the Prior network
    Args:
        X - Input tensor in nhwc format
        state - state from previous layer
        kernel_shape - height and width of kernel
        name - name for scoping
    Returns:
        gated output, and layer state
    """

    # refactored this to implement to Fig.2 in https://arxiv.org/pdf/1606.05328.pdf
    # rather than trying to implement eq. 4/5
    with tf.variable_scope(name):
        batch_size, height, width, in_channel = X.get_shape().as_list()
        kernel_h, kernel_w = kernel_shape

        # left side / state input to layer
        left = conv_layer(state, 2 * in_channel, kernel_shape, mask_type='c', name='left_conv')
        # helper function to split in two and apply tanh and sigmoind
        new_state = split_and_gate(left, in_channel)

        # convolution from left side to right side. state -> output
        left_to_right_conv = conv_layer(left, 2 * in_channel, [1, 1], name="middle_conv")

        # right side / output
        right = conv_layer(X, 2 * in_channel, [1, kernel_w], mask_type='b', name='right_conv1')
        right = right + left_to_right_conv
        new_output = split_and_gate(right, in_channel)
        new_output = conv_layer(new_output, in_channel, [1, 1], mask_type='b', name='right_conv2')
        new_output = new_output + X

        return new_output, new_state


def conv_layer(X, out_channels, kernel_shape, strides=[1, 1], mask_type=None, name=None):
    '''
    Convolutional layer capable of being masked
    Args:
        X - input tensor in nhwc format
        out_channels - number of output channels to use
        kernel_shape - height and width of kernel
        strides - stride size
        mask_type - type of mask to use. Masks using one of the following A/B/vertical stack mask from https://arxiv.org/pdf/1606.05328.pdf
        name - name for scoping
    Returns:
        2d convolution layer
    '''
    # refactored get_weights and conv_layer into one function so it is much simpler and less fragile
    with tf.variable_scope(name):
        kernel_h, kernel_w = kernel_shape
        batch_size, height, width, in_channel = X.get_shape().as_list()

        # center coords of kernel/mask
        center_h = kernel_h // 2
        center_w = kernel_w // 2

        if mask_type:
            # using zeros is easier than ones, because horizontal stack
            mask = np.zeros((kernel_h, kernel_w, in_channel, out_channels), dtype=np.float32)

            # vertical stack only, no horizontal stack
            mask[:center_h, :, :, :] = 1

            if mask_type == 'a':  # no center pixel in mask
                mask[center_h, :center_w, :, :] = 0
            elif mask_type == 'b':  # center pixel in mask
                mask[center_h, :center_w + 1, :, :] = 1
        else:
            # no mask
            mask = np.ones((kernel_h, kernel_w, in_channel, out_channels), dtype=np.float32)

        # initialize and mask weights
        weights_shape = [kernel_h, kernel_w, in_channel, out_channels]

        # need to experiment with truncated normal vs xavier glorot
        # weights_initializer = tf.contrib.layers.xavier_initializer()
        weights_initializer = tf.truncated_normal_initializer(stddev=0.1)

        weights = tf.get_variable("weights", shape=weights_shape,
                                  dtype=tf.float32, initializer=weights_initializer)
        weights = weights * mask

        bias = tf.get_variable('bias', shape=[out_channels],
                               dtype=tf.float32, initializer=tf.constant_initializer(0.0))

        output = tf.nn.conv2d(X, weights, [1, strides[0], strides[1], 1], padding="SAME")
        output = tf.nn.bias_add(output, bias)

    return output


def residual_block(X, out_channels, kernel_shape, strides=[1, 1], name=None):
    '''
    ResNet block from https://arxiv.org/pdf/1512.03385.pdf
    Args:
        X - input tensor in nhwc format
        out_channels - number of output channels to use
        kernel_shape - height and width of kernel
        strides - stride size
        name - name for scoping
    '''
    with tf.variable_scope(name):
        conv1 = conv_layer(X, out_channels, kernel_shape, strides=strides, name="conv1")
        batch_norm1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True, epsilon=1e-8)
        relu1 = tf.nn.relu(batch_norm1)

        conv2 = conv_layer(relu1, out_channels, kernel_shape, strides=strides, name="conv2")
        batch_norm2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, epsilon=1e-8)

        output = X + batch_norm2
        return output


def transposed_conv2d_layer(X, out_channels, kernel_shape, strides=[1, 1], name="deconv2d"):
    """
    tranposed convolution layer
    Args:
      X - input tensor in nhwc format
      out_channels - number of output channels to use
      kernel_shape - height and width of kernel
      strides - stride size
      name - name for scoping
    Returns:
        upscaled image tensor in nhwc format
    """
    # I was making this way too complicated before. much simpler now
    with tf.variable_scope(name):

        # need to experiment with truncated_normal vs xavier glorot initialization
        # weights_initializer = tf.contrib.layers.xavier_initializer()
        weights_initializer = tf.truncated_normal_initializer(stddev=0.1)

        return tf.contrib.layers.convolution2d_transpose(X, out_channels, kernel_shape, strides,
                                                         padding='SAME', weights_initializer=weights_initializer,
                                                         biases_initializer=tf.constant_initializer(0.0))
