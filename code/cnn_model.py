# -*- coding: utf-8 -*-
"""
author : Robert Yin
"""
import tensorflow as tf


class CNN_NET(object):
    def __init__(self, sequence_length, embedding_size,
                 filter_sizes, num_filters,
                 batch_size, l2_reg_lambda=0.0):
        self.label = tf.placeholder(tf.int32, [None, 2], name="label")
        self.embedded_chars = tf.placeholder(tf.float32,
                                             [None, sequence_length, embedding_size],
                                             name="embedded_chars")
        self.dropout_keep_prob = tf.placeholder(tf.float32,
                                                name="dropout_keep_prob")

        embedded_chars_expand = tf.expand_dims(self.embedded_chars, -1)

        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % str(filter_size)):
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    embedded_chars_expand,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                h = tf.nn.relu(tf.nn.bias_add(conv, b))
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length - filter_size + 1, 1, 1],
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="poll"
                )
                pooled_outputs.append(pooled)

        num_filters_total = num_filters * len(filter_sizes)
        pooled_flat = tf.reshape(tf.concat(pooled_outputs, 3),
                                 [-1, num_filters_total])

        with tf.name_scope("fully-connected-1"):
            FC_shape1 = [num_filters_total, 100]
            W1 = tf.Variable(tf.truncated_normal(FC_shape1, stddev=0.1), name="W")
            b1 = tf.Variable(tf.constant(0.1, shape=[100]), name="b")
            h_tmp1 = tf.nn.bias_add(tf.matmul(pooled_flat, W1), b1)
            h1_real = tf.nn.relu(h_tmp1)
            h1 = tf.nn.dropout(h1_real, self.dropout_keep_prob)

        with tf.name_scope("fully-connected-2"):
            FC_shape2 = [100, 2]
            W2 = tf.Variable(tf.truncated_normal(FC_shape2, stddev=0.1), name="W")
            b2 = tf.Variable(tf.constant(0.1, shape=[2]), name="b")
            h_tmp2 = tf.nn.bias_add(tf.matmul(h1, W2), b2)
            h2_real = tf.nn.relu(h_tmp2)

        logits = tf.nn.softmax_cross_entropy_with_logits(logits=h2_real, labels=self.label)
        self.loss = tf.reduce_mean(logits)  # cross_entropy
        self.prediction = tf.argmax(h2_real, 1)
        self.correct_prediction = tf.equal(self.prediction, tf.argmax(self.label, 1))
        self.accuracy = tf.reduce_mean(tf.cast(self.correct_prediction, tf.float32))


