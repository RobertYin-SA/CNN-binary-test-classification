This is a basic CNN network demo on Spam message classification(binary classification) model.

An implementation of [Convolutional Neural Networks for Sentence Classification](http://arxiv.org/abs/1408.5882) in Spam messages indentifying(make a little change to prevent overfitting).

## Requirements

- Python 2.7
- Tensorflow1.0
- Numpy
- gensim

## Running step(please run the next script in code/)
```
python gen_split_words.py --input_dir ../data/whole.txt --output_dir ../data/whole_split_words.txt
```

```
cat ../data/whole_split_words.txt | python ../gen_wordvec/preprocess.py > ../gen_wordvec/wordvec_data_prepare
```

```
python ../gen_wordvec/train_word2vec_model.py ../gen_wordvec/wordvec_data_prepare ../gen_wordvec/word2vec_model ../gen_wordvec/word2vec_vector
```

```
cat ../data/whole_split_words.txt | python split_file.py ../data/train_split_words.txt ../data/test_split_words.txt
```

```
python cnn_train.py > pro 
# When training, there are also output precision file in testing file, which will help us to check performance.
```

```
python cnn_test.py 
# If you run a new model, please change the model_path !!
```

```
python evaluation.py ../data/test_split_words.txt precision # computing the acc and recall
```
