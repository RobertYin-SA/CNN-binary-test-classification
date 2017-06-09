# -*- coding: utf-8 -*-
import sys
import random

with open(sys.argv[1], 'w') as train_file, open(sys.argv[2], 'w') as test_file:
    for line in sys.stdin:
        if random.random() < 0.5:
            train_file.write(line)
        else:
            test_file.write(line)

