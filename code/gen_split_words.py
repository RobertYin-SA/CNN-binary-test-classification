# -*- coding: utf-8 -*-
"""
author : Robert Yin

date : 2017/05/27
usage :
    python gen_split_words.py --input_dir ../data/whole.txt --output_dir ../data/whole_split_words.txt
"""
import argparse
import jieba


def main(args):
    num_bugs = 0
    with open(args.output_dir, 'w') as output_file:
        with open(args.input_dir, 'r') as input_file:
            for line in input_file:
                try:
                    _, label, sentence = line.decode('utf-8').strip().split('\t')
                except Exception:
                    num_bugs += 1
                    continue
                sentence_words = list(jieba.cut(sentence))
                new_sentence = ' '.join(sentence_words)
                output_file.write((label + '\t' + new_sentence + '\n').encode('utf-8'))

    print 'There are %s bugs' % str(num_bugs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Specify arguments')
    parser.add_argument('--input_dir', help='Original classification text file path')
    parser.add_argument('--output_dir', help='Output split words classification text file path')
    args = parser.parse_args()
    main(args)
