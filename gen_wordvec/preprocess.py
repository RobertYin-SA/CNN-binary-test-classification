# -*- coding: utf-8 -*-
import sys
for line in sys.stdin:
    _, sentence = line.strip('\n').split('\t')
    print sentence
