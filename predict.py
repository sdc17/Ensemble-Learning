import os
import sys
import ast
import math
import glob
import random
import argparse
import datetime
import numpy as np
import matplotlib.pyplot as plt
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from concurrent.futures import ProcessPoolExecutor, Executor, as_completed


class Predictor:
    def __init__(self, operation):
        self.operation = operation
        if operation == 'valid':
            try:
                self.data_valid = np.load('./dataset/data_valid.npy', allow_pickle=True)
                self.label_valid = np.load('./dataset/label_valid.npy', allow_pickle=True)
                self.contents = len(self.label_valid) 
            except:
                print("Error: Dataset path does not exist!")
                return
        elif operation == 'test':
            try:
                self.data_test = np.load('./dataset/data_test.npy', allow_pickle=True)
                self.contents = len(self.data_test) 
            except:
                print("Error: Dataset path does not exist!")
                return

    def set_params(self, backbone, ensemble, hyper):
        self.backbone = backbone
        self.ensemble = ensemble
        self.hyper = hyper

    def rmse(self, label, pred):
        return math.sqrt(mean_squared_error(label, pred))
    
    def task(self, i):
        classifier = joblib.load(os.path.join('model', self.ensemble, self.backbone + '_' + str(i) + '.pkl'))
        if self.operation == 'valid':
            pred = classifier.predict(self.data_valid)
        elif self.operation == 'test':
            pred = classifier.predict(self.data_test)
        return pred

    def valid(self):
        if self.ensemble == 'bagging':
            preds = []
            with ProcessPoolExecutor() as executor:
                for results in executor.map(self.task, range(self.hyper)):
                    preds.append(results)
            results = []
            for i in range(self.contents):
                result_list = [preds[x][i] for x in range(self.hyper)]
                counts = np.bincount(result_list)
                # results.append(np.argmax(counts))
                res = 0
                counts = np.array(counts, 'float')
                counts /= sum(counts)
                for j in range(len(counts)):
                    res += j * counts[j]
                results.append(res)
        elif self.ensemble == 'adaboost':
            betas = np.load(os.path.join('model', self.ensemble, self.backbone + '_' + 'beta.npy'), allow_pickle=True)
            length = min(len(betas), self.hyper)
            preds = []
            with ProcessPoolExecutor() as executor:
                for results in executor.map(self.task, range(length)):
                    preds.append(results)
            results = []
            for i in range(self.contents):
                label = np.zeros(6)
                for x in range(length):
                    label[preds[x][i]] += math.log(1 / betas[x])
                # results.append(np.argmax(label))    
                res = 0
                counts = np.array(label, 'float')
                counts /= sum(counts)
                for j in range(len(counts)):
                    res += j * counts[j]
                results.append(res) 
        results = np.array(results)
        rmse_value = -1
        if self.operation == 'valid':
            rmse_value = self.rmse(self.label_valid, results)
            print("Mean acc: {} RMSE: {}".format(np.mean(self.label_valid == results), rmse_value))
        return results, rmse_value

    def test(self):
        results, _ = self.valid()
        output = ['id,predicted\n'] + [str(i + 1) + ',' + str(float(results[i])) + '\n' for i in range(self.contents)]
        with open('./data/output.csv', 'w') as f:
            f.writelines(output)

    def predict(self):
        if self.operation == 'valid':
            _, rmse_value = self.valid()
            return rmse_value
        elif self.operation == 'test':
            self.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',type=str, choices=['svm', 'dtree', 'knn', 'bayes'],required=False, help="Backbone to train")
    parser.add_argument('--ensemble',type=str, choices=['bagging','adaboost'],required=False, help="Ensemble way to train")
    parser.add_argument('--operation',type=str, choices=['valid','test'],required=False, help="Operation to take")
    parser.add_argument('--hyper',type=int, required=False, help="Hyperparameter for train")
    parser.add_argument("--analysis", default=False, type=ast.literal_eval, choices=[True, False], help="Analysis function")
    args = parser.parse_args()

    if not args.analysis:
        print('===> Predicting <===')
        start = datetime.datetime.now()
        predictor = Predictor(args.operation)
        predictor.set_params(args.backbone, args.ensemble, args.hyper)
        predictor.predict()
        end = datetime.datetime.now()
        print('Time cost: {}'.format(end -start))
        print('===> Completed! <===')
        print('-' * 40)
    else:
        print('===> Analysis function <===')
        start = datetime.datetime.now()
        if not os.path.exists('./analysis'):
            os.mkdir('./analysis')
        bagging_svm, bagging_dtree, adaboost_svm, adaboost_dtree = [], [], [], []
        predictor = Predictor('valid')
        for i in range(11):
            predictor.set_params('svm', 'bagging', 2**i)
            bagging_svm.append(predictor.predict())
            predictor.set_params('dtree', 'bagging', 2**i)
            bagging_dtree.append(predictor.predict())
            predictor.set_params('svm', 'adaboost', 2**i)
            adaboost_svm.append(predictor.predict())
            predictor.set_params('dtree', 'adaboost', 2**i)
            adaboost_dtree.append(predictor.predict())
        axes = [str(2**i) for i in range(11)]
        plt.figure(figsize=(16, 10))
        plt.plot(axes, bagging_svm, color='royalblue', label='bagging with svms')
        plt.plot(axes, bagging_dtree, color='cyan', label='bagging with dtrees')
        plt.plot(axes, adaboost_svm, color='coral', label='adaboost with svms')
        plt.plot(axes, adaboost_dtree, color='palegreen', label='adaboost with dtrees')
        # plt.xticks(axes)
        # plt.yticks(np.arr)
        plt.xlabel("Number of Weak classifiers", fontsize=14)
        plt.ylabel("RMSE", fontsize=14)
        plt.legend()
        plt.savefig('./analysis/rmse_with_weak_classifiers.png')
        print("RMSE for bagging with {} svms: {:.4f}".format(1024, bagging_svm[10]))
        print("RMSE for bagging with {} dtrees: {:.4f}".format(1024, bagging_dtree[10]))
        print("RMSE for adaboost with {} svms: {:.4f}".format(1024, adaboost_svm[10]))
        print("RMSE for adaboost with {} dtrees: {:.4f}".format(1024, adaboost_dtree[10]))
        end = datetime.datetime.now()
        print('Time cost: {}'.format(end -start))
        print('===> Completed! <===')
        print('-' * 40)


