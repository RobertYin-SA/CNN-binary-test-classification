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
tf.flags.DEFINE_integer("evaluate_every", 3000, "Evaluate model on dev set after this many steps")
tf.flags.DEFINE_integer("checkpoint_every", 3000, "Save model after this many steps")

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

    # Define Training procedure
    global_step = tf.Variable(0, name="global_step", trainable=False)
    optimizer = tf.train.AdamOptimizer(1e-4)
    grads_and_vars = optimizer.compute_gradients(cnn.loss)
    train_op = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

    # Keep track of gradient values and sparsity (optional)
    grad_summaries = []
    for g, v in grads_and_vars:
        if g is not None:
            grad_hist_summary = tf.summary.histogram("{}/grad/hist".format(v.name), g)
            sparsity_summary = tf.summary.scalar("{}/grad/sparsity".format(v.name), tf.nn.zero_fraction(g))
            grad_summaries.append(grad_hist_summary)
            grad_summaries.append(sparsity_summary)
    grad_summaries_merged = tf.summary.merge(grad_summaries)

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Output directory for models and summaries
    timestamp = str(int(time.time()))
    out_dir = os.path.abspath(os.path.join(os.path.curdir, "runs", timestamp))
    print("Writing to {}\n".format(out_dir))

    # Summaries for loss and accuracy
    loss_summary = tf.summary.scalar("loss", cnn.loss)
    # acc_summary = tf.scalar_summary("accuracy", cnn.accuracy)
    acc_summary = tf.summary.scalar("accuracy", cnn.accuracy)

    # Train Summaries
    train_summary_op = tf.summary.merge([loss_summary, acc_summary, grad_summaries_merged])
    train_summary_dir = os.path.join(out_dir, "summaries", "train")
    # train_summary_writer = tf.train.SummaryWriter(train_summary_dir, sess.graph_def)
    train_summary_writer = tf.summary.FileWriter(train_summary_dir, sess.graph_def)

    checkpoint_dir = os.path.abspath(os.path.join(out_dir, "checkpoints"))
    checkpoint_prefix = os.path.join(checkpoint_dir, "model")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    saver = tf.train.Saver(tf.global_variables())

    # Initialize all variables
    sess.run(tf.global_variables_initializer())

    def train_step(batch_labels, batch_embedded_chars):
        feed_dict = {
            cnn.label: batch_labels,
            cnn.embedded_chars: batch_embedded_chars,
            cnn.dropout_keep_prob: FLAGS.dropout_keep_prob
        }

        _, step, summaries, loss, accuracy = sess.run(
            [train_op, global_step, train_summary_op,
             cnn.loss, cnn.accuracy], feed_dict)
        time_str = datetime.datetime.now().isoformat()
        print("{}: step {}, loss {:g}, acc {:g}".format(time_str, step, loss, accuracy))
        train_summary_writer.add_summary(summaries, step)

    """
    def dev_step():
        N_acc = []
        acc = []

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

            accuracy = sess.run([cnn.accuracy], feed_dict)
            acc.append(accuracy[0])

        acc_all = sum(np.array(acc) * np.array(N_acc)) / sum(N_acc)

        return acc_all
    """
    def dev_step(prediction_file):
        N_acc = []
        acc = []

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



    for i in xrange(FLAGS.num_epochs):
        print "==================="
        print "epoch:" + str(i)
        print "==================="
        for (batch_labels, batch_embedded_chars) in\
                gen_batch.gen_batch_all(train_file_path,
                                        word_emb=word_emb,
                                        sequence_len=sequence_length,
                                        embedding_size=FLAGS.embedding_size,
                                        batch_size=FLAGS.batch_size):
            try:
                train_step(batch_labels, batch_embedded_chars)
                current_step = tf.train.global_step(sess, global_step)
                if current_step % FLAGS.evaluate_every == 0:
                    print("\nEvaluation:")
                    """
                    dev_acc = dev_step()
                    """
                    pred_file = open("precision", 'w')
                    dev_step(pred_file)
                    pred_file.close()
                    # print "The acc in testing dataset is %s" % dev_acc
                    print("")
            except Exception as e:
                print (e)

            if current_step % FLAGS.checkpoint_every == 0:
                path = saver.save(sess, checkpoint_prefix, global_step=current_step)
                print("Saved model checkpoint to {}\n".format(path))

