# -*- coding: utf-8 -*-
"""
author : Robert Yin
"""
import numpy as np


def get_maxlength(docfile_path):
    max_len = 0
    with open(docfile_path, 'r') as docfile:
        for line in docfile:
            _, sentence = line.strip().split('\t')
            sent_len = len(sentence.split())
            if sent_len > max_len:
                max_len = sent_len
    return max_len


def gen_word_emb(word_emb_file_path):
    word_emb = {}
    with open(word_emb_file_path, 'r') as word_emb_file:
        for line in word_emb_file:
            line = line.decode('utf-8').strip()
            mes = line.split()
            word = mes[0]
            emb = map(lambda x: float(x), mes[1:])
            word_emb[word] = emb
    return word_emb


def cropORpad(sequence, maxlen, NILWORD_emb):
    pad_num = maxlen - len(sequence)
    if pad_num > 0:
        for i in xrange(pad_num):
            sequence.append(NILWORD_emb)
    else:
        for i in xrange(-pad_num):
            sequence.pop()
    return sequence


def encoder(sequence_word, word_emb, NILWORD_emb):
    code = []
    for word in sequence_word:
        code.append(word_emb.get(word, NILWORD_emb))
    return code


def reshape_batch(batches):
    batch_labels = []
    batch_embedded_chars = []
    for batch in batches:
        batch_labels.append(batch['label'])
        batch_embedded_chars.append(batch['sentence_emb'])
    return (np.array(batch_labels), np.array(batch_embedded_chars))


def gen_batch_all(train_file_path, word_emb, sequence_len,
                  embedding_size, batch_size):
    NILWORD_emb = np.zeros(embedding_size)
    batches = []
    count = 0
    with open(train_file_path, 'r') as train_file:
        for line in train_file:
            count += 1
            line = line.decode('utf-8').strip()
            label, sentence = line.split('\t')
            if label == '1':
                label = [0, 1]
            else:
                label = [1, 0]
            sentence_word = sentence.split()
            sentence_emb_tmp = encoder(sentence_word, word_emb, NILWORD_emb)
            sentence_emb = cropORpad(sentence_emb_tmp, sequence_len, NILWORD_emb)
            batch = {}
            batch['label'] = label
            batch['sentence_emb'] = np.array(sentence_emb)
            batches.append(batch)
            if count % batch_size == 0:
                yield reshape_batch(batches)
                batches = []
        if len(batches) != 0:
            yield reshape_batch(batches)


