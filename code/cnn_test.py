# -*- coding: utf-8 -*-
"""
author : Robert Yin
"""
import tensorflow as tf
import numpy as np
import time
import os
import datetime
import cnn_model
import gen_batch
from tqdm import tqdm

# Parameters

# Model Hyperparameters
tf.flags.DEFINE_integer("embedding_size", 400, "Dimensionality of character embedding")
tf.flags.DEFINE_string("filter_sizes", "1,2,3,5,8,15", "Comma-separated filter sizes")
tf.flags.DEFINE_integer("num_filters", 500, "Number of filters per filter size")
tf.flags.DEFINE_float("dropout_keep_prob", 0.5, "Dropout keep probability")
tf.flags.DEFINE_float("l2_reg_lambda", 0.0, "L2 regularizaion lambda (default: 0.0)")

# Training parameters
tf.flags.DEFINE_integer("batch_size", 100, "Batch Size")
tf.flags.DEFINE_integer("num_epochs", 200, "Number of training epochs")
tf.flags.DEFINE_integer("evaluate_every", 500, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 500, "Save model after this many steps")

FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
print("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    print("{}={}".format(attr.upper(), value))
print("")

# Data Preparation
train_file_path = "../data/train_split_words.txt"
test_file_path = "../data/test_split_words.txt"
word_emb_file_path = "../gen_wordvec/word2vec_vector"
# If you run a new model, please change the model_path
model_path = "runs/1496860553/checkpoints/model-9000"
word_emb = gen_batch.gen_word_emb(word_emb_file_path)
sequence_length = gen_batch.get_maxlength(train_file_path)


sess = tf.Session()
with sess.as_default():
    cnn = cnn_model.CNN_NET(sequence_length=sequence_length,
                            embedding_size=FLAGS.embedding_size,
                            filter_sizes=map(lambda x: int(x), FLAGS.filter_sizes.split(',')),
                            num_filters=FLAGS.num_filters,
                            batch_size=FLAGS.batch_size,
                            l2_reg_lambda=FLAGS.l2_reg_lambda)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

    saver = tf.train.Saver(tf.global_variables())

    saver.restore(sess, model_path)

    def dev_step(prediction_file):
        N_acc = []

        for (batch_labels, batch_embedded_chars) in\
                tqdm(gen_batch.gen_batch_all(test_file_path,
                                             word_emb=word_emb,
                                             sequence_len=sequence_length,
                                             embedding_size=FLAGS.embedding_size,
                                             batch_size=FLAGS.batch_size)):
            N_acc.append(batch_labels.shape[0])

            feed_dict = {
                cnn.label: batch_labels,
                cnn.embedded_chars: batch_embedded_chars,
                cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
            }

            prediction = sess.run([cnn.prediction], feed_dict)
            prediction = prediction[0]

            for i in xrange(len(prediction)):
                prediction_file.write(str(prediction[i]) + '\n')

    pred_file = open("precision", 'w')
    dev_step(pred_file)
    pred_file.close()


