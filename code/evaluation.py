# -*- coding: utf-8 -*-
"""
usage:
    python evaluation.py ../data/test_split_words.txt precision
"""
import sys
import numpy as np

real_test_label = []
with open(sys.argv[1], 'r') as test_file:
    for line in test_file:
        label, _ = line.split('\t')
        real_test_label.append(int(label))

pred_test_label = []
with open(sys.argv[2], 'r') as pred_test_file:
    for line in pred_test_file:
        label = line.strip()
        pred_test_label.append(int(label))


correct_pos = 0
correct_neg = 0
count_all = 0
count_pos_precision = 0
count_neg_precision = 0
count_pos_recall = 0
count_neg_recall = 0
for i in xrange(len(real_test_label)):
    count_all += 1
    if real_test_label[i] == 1:
        count_pos_recall += 1

    if real_test_label[i] == 0:
        count_neg_recall += 1

    if pred_test_label[i] == 1:
        count_pos_precision += 1

    if pred_test_label[i] == 0:
        count_neg_precision += 1

    if real_test_label[i] == pred_test_label[i] and real_test_label[i] == 1:
        correct_pos += 1

    if real_test_label[i] == pred_test_label[i] and real_test_label[i] == 0:
        correct_neg += 1

print "===================================="
print "ALL PRECISION = %s" % str((correct_pos + correct_neg) / float(count_all))
print "===================================="
print "ACC:"
print "1 acc = %s" % str(correct_pos / float(count_pos_precision))
print "0 acc = %s" % str(correct_neg / float(count_neg_precision))
print "===================================="
print "RECALL:"
print "1 recall = %s" % str(correct_pos / float(count_pos_recall))
print "0 recall = %s" % str(correct_neg / float(count_neg_recall))
print "===================================="

