import os
import re
import sys
import ast
import random
import argparse
import datetime
import pandas as pd
import numpy as np
from stopwords import stops
from gensim.models import Word2Vec, KeyedVectors
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import scale
from concurrent.futures import ProcessPoolExecutor, Executor, as_completed
# from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer


def extract_words(sentence, pattern):
    feature_words = re.findall(pattern, sentence.lower())
    feature_words = [w for w in feature_words if w not in stops]
    return feature_words


def divide_dataset_task(task_list):
    data, label = [], []
    # regex = r'[a-zA-Z0-9]+'
    regex = r'[a-zA-Z]+'
    pattern = re.compile(regex)
    for value in task_list:
        # data.append(extract_words(str(value['summary']) + ' ' + str(value['reviewText']) + \
        #     ' ' + ' '.join([str(value['reviewerID'])] * 2) + ' ' + ' '.join([str(value['asin'])]*4), pattern))
        data.append(extract_words(str(value['summary']) + ' ' + str(value['reviewText']), pattern) + \
            [str(value['reviewerID'])] * 0 + [str(value['asin'])] * 0)
        label.append(int(value['overall']))
    return {'data': data, 'label': label}    


def divide_testset_task(task_list):
    data = {}
    # regex = r'[a-zA-Z0-9]+'
    regex = r'[a-zA-Z]+'
    pattern = re.compile(regex)
    for key, value in task_list:
        # data[key] = extract_words(str(value['summary']) + ' ' + str(value['reviewText']) + \
        #     ' ' + ' '.join([str(value['reviewerID'])] * 2) + ' ' + ' '.join([str(value['asin'])]*4), pattern)
        data[key] = extract_words(str(value['summary']) + ' ' + str(value['reviewText']), pattern) + \
            [str(value['reviewerID'])] * 0 + [str(value['asin'])] * 0
    return data


def compute_tfidf(data_train, data_valid, data_test):
    data_train = [' '.join(i) for i in data_train]
    data_valid = [' '.join(i) for i in data_valid]
    data_test = [' '.join(i) for i in data_test]
    data = data_train + data_valid + data_test

    counter_data = CountVectorizer(min_df=5)
    count_data = counter_data.fit_transform(data)
    counter_train = CountVectorizer(vocabulary=counter_data.vocabulary_)
    count_train = counter_train.fit_transform(data_train)
    counter_valid = CountVectorizer(vocabulary=counter_data.vocabulary_)
    count_valid = counter_valid.fit_transform(data_valid)
    counter_test = CountVectorizer(vocabulary=counter_data.vocabulary_)
    count_test = counter_test.fit_transform(data_test)

    print("Shape of train: {} valid:{} test:{}".format(count_train.shape, count_valid.shape, count_test.shape))

    tfidf = TfidfTransformer()
    np.save('/dataset/data_train.npy', tfidf.fit_transform(count_train).toarray())
    np.save('/dataset/data_valid.npy', tfidf.fit_transform(count_valid).toarray())
    np.save('/dataset/data_test.npy', tfidf.fit_transform(count_test).toarray())


def compute_word2vec(data_train, data_valid, data_test, dim, need_train, need_PCA=False, n_components=128):
    
    data = data_train + data_valid + data_test
    print('===> Training/Loading word2vec model')
    start = datetime.datetime.now()
    if need_train:
        model = Word2Vec(data, sg=1, size=dim, window=5, min_count=5, workers=16, iter=10)
        if not os.path.exists('model'):
            os.makedirs('./model')
        model.save('./model/word2vec.bin')
    else:
        model = Word2Vec.load('./model/word2vec.bin')
        # model = KeyedVectors.load_word2vec_format('./model/word2vec.txt', binary = False)
        # model = KeyedVectors.load_word2vec_format('./data/GoogleNews-vectors-negative300.bin', binary = True)
    end = datetime.datetime.now()
    print('Time cost: {}'.format(end -start))
    print('===> Completed!')
    print('-' * 20)

    def sen2vec(sentence, size):
        sens = np.zeros(size).reshape((1, size))
        cnt = 0
        for word in sentence:
            try:
                sens += model[word].reshape((1, size))
                cnt += 1
            except:
                continue
        if cnt > 0:
            sens /= cnt
        return sens
    
    data_train = scale(np.concatenate([sen2vec(sen, dim) for sen in data_train]))
    data_valid = scale(np.concatenate([sen2vec(sen, dim) for sen in data_valid]))
    data_test = scale(np.concatenate([sen2vec(sen, dim) for sen in data_test]))

    if need_PCA:
        pca=PCA(n_components=n_components)
        pca.fit(data_train)
        data_train = pca.transform(data_train)
        data_valid = pca.transform(data_valid)
        data_test = pca.transform(data_test)
        print("666")

    np.save('./dataset/data_train.npy', data_train)
    np.save('./dataset/data_valid.npy', data_valid)
    np.save('./dataset/data_test.npy', data_test)


def divide_dataset(valid, dim, need_train):

    print('===> Processing train.csv')
    start = datetime.datetime.now()
    train_df = pd.read_csv('./data/train.csv', sep='\t')
    train_dict = train_df.to_dict(orient='index')
    train_dict_apart = np.array_split(list(train_dict.values()), 16)
    data, label = [], []
    with ProcessPoolExecutor() as executor:
        for results in executor.map(divide_dataset_task, train_dict_apart):
            data.extend(results['data'])
            label.extend(results['label'])
    data_train, data_valid, label_train, label_valid = train_test_split(data, label, test_size=valid)
    end = datetime.datetime.now()
    print('Time cost: {}'.format(end -start))
    print('===> Completed!')
    print('-' * 20)

    print('===> Processing test.csv')
    start = datetime.datetime.now()
    data_test = {}
    test_df = pd.read_csv('./data/test.csv', sep='\t')
    test_dict = test_df.to_dict(orient='index')
    test_dict_apart = np.array_split(list(test_dict.items()), 16)
    with ProcessPoolExecutor() as executor:
        for results in executor.map(divide_testset_task, test_dict_apart):
            data_test.update(results)
    data_test = list(dict(sorted(data_test.items(), key = lambda x: x[0], reverse = False)).values())
    end = datetime.datetime.now()
    print('Time cost: {}'.format(end -start))
    print('===> Completed!')
    print('-' * 20)

    print('===> Computing word2vec')
    start = datetime.datetime.now()
    if not os.path.exists('./dataset'):
        os.makedirs('./dataset')
    # compute_tfidf(data_train, data_valid, data_test)
    compute_word2vec(data_train, data_valid, data_test, dim=dim, need_train=need_train)
    print('Time cost: {}'.format(end -start))
    print('===> Completed!')
    print('-' * 20)

    np.save('./dataset/label_train.npy', label_train)
    np.save('./dataset/label_valid.npy', label_valid)
    # np.save('./dataset/raw_test.npy', data_test)
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid", default=0.05, type=float, help="Ratio of validation set")
    parser.add_argument("--dim", default=128, type=int, help="Dimension of word vector")
    parser.add_argument("--need_train", default=True, type=ast.literal_eval, choices=[True, False], help="Whether to train word2vec or not")
    args = parser.parse_args()

    print('===> Dividing <===')
    start = datetime.datetime.now()
    # extract_words("Works well for animals that chew wires....	Last year, \
    #     I rescued a half-dead stray kitten and spent a bundle of money on vet bills to save her.  ")
    divide_dataset(valid=args.valid, dim=args.dim, need_train=args.need_train)
    end = datetime.datetime.now()
    print('Time cost: {}'.format(end -start))
    print('===> Completed! <===')
    print('-' * 40)
    