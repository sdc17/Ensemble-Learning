
import os
import sys
import random
import argparse
import datetime
import numpy as np
from sklearn import tree
from sklearn.externals import joblib
from sklearn.decomposition import PCA
from sklearn.svm import SVC, LinearSVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier as KNN
from concurrent.futures import ProcessPoolExecutor, Executor, as_completed


class Trainer:
    def __init__(self):
        try:
            self.data_train = np.load('./dataset/data_train.npy', allow_pickle=True)
            self.label_train = np.load('./dataset/label_train.npy', allow_pickle=True)
            self.data_valid = np.load('./dataset/data_valid.npy', allow_pickle=True)
            self.label_valid = np.load('./dataset/label_valid.npy', allow_pickle=True)
            self.contents = len(self.label_train) 
        except:
            print("Error: Dataset path does not exist!")
            return

    def set_params(self, backbone, ensemble, hyper):
        self.backbone = backbone
        self.ensemble = ensemble
        self.hyper = hyper

    def train_backbone(self, i, data_train_resample, label_train_resample, weights=None):
        
        if self.backbone == 'svm':
            classifier = LinearSVC(multi_class='ovr', class_weight='balanced', max_iter=60)
            # classifier = SVC(kernel='rbf', decision_function_shape='ovo', class_weight='balanced', cache_size=800)
        elif self.backbone == 'dtree':
            # classifier = tree.DecisionTreeClassifier(class_weight='balanced')
            classifier = tree.DecisionTreeClassifier(min_samples_leaf=5)
        elif self.backbone == 'knn':
            classifier = KNN(n_neighbors=5, weights='distance')
        elif self.backbone == 'bayes':
            classifier == GaussianNB()

        classifier.fit(data_train_resample, label_train_resample)
        print("Score of No.{} {}: {}".format(i, self.backbone, classifier.score(self.data_valid, self.label_valid)))

        if not os.path.exists(os.path.join('model', self.ensemble)):
            os.mkdir(os.path.join('model', self.ensemble))

        if self.ensemble == 'bagging':
            joblib.dump(classifier, os.path.join('model', self.ensemble, self.backbone + '_' + str(i) + '.pkl'))

        elif self.ensemble == 'adaboost':
            pred = classifier.predict(self.data_train)
            epsilon = float(np.dot(np.array(pred) != np.array(self.label_train), weights))
            if epsilon > 0.5:
                return epsilon, weights
            beta = epsilon / (1 - epsilon)
            co = [1.0 if pred[i] != self.label_train[i] else beta for i in range(self.contents)]
            weights = np.multiply(co, weights)
            weights /= np.sum(weights)
            joblib.dump(classifier, os.path.join('model', self.ensemble, self.backbone + '_' + str(i) + '.pkl'))
            if not os.path.exists(os.path.join('model', self.ensemble, self.backbone + '_' + 'beta.npy')):
                np.save(os.path.join('model', self.ensemble, self.backbone + '_' + 'beta.npy'), np.array([beta]))
            else:
                betas = np.load(os.path.join('model', self.ensemble, self.backbone + '_' + 'beta.npy'), allow_pickle=True)
                beta_list = list(betas)
                beta_list.append(beta)
                np.save(os.path.join('model', self.ensemble, self.backbone + '_' + 'beta.npy'), np.array(beta_list))
            return epsilon, weights

    def train_bagging_task(self, i):
        print("Training No.{} {} in {} ".format(i, self.backbone, self.ensemble))
        data_train_resample, label_train_resample = [], []
        np.random.seed(i * 114514)
        index = np.random.choice(self.contents, size=self.contents)
        data_train_resample = self.data_train[index]
        label_train_resample = self.label_train[index]
        # self.train_backbone(i, np.array(data_train_resample), np.array(label_train_resample))   
        self.train_backbone(i, data_train_resample, label_train_resample) 
        print("Training No.{} {} in {} Completed!".format(i, self.backbone, self.ensemble))

    def train(self):
        if self.ensemble == 'bagging':
            with ProcessPoolExecutor() as executor:
                executor.map(self.train_bagging_task, range(self.hyper))
        elif self.ensemble == 'adaboost':
            weights = np.array([1.0 / self.contents] * self.contents)
            if os.path.exists(os.path.join('model', self.ensemble, self.backbone + '_' + 'beta.npy')):
                os.remove(os.path.join('model', self.ensemble, self.backbone + '_' + 'beta.npy')) 
            i = 0
            while i < self.hyper:
            # for i in range(self.hyper):
                print("Training No.{} {} in {} ".format(i, self.backbone, self.ensemble))
                random.seed(i * 114514)
                index = np.random.choice(self.contents, size=self.contents, p=weights)
                data_train_resample = self.data_train[index]
                label_train_resample = self.label_train[index]
                epsilon, weights = self.train_backbone(i, data_train_resample, label_train_resample, weights)
                if epsilon > 0.5:
                    # break
                    weights = np.array([1.0 / self.contents] * self.contents)
                    print("Break on {}".format(i))
                else:
                    i += 1



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--backbone',type=str, choices=['svm', 'dtree', 'knn', 'bayes'],required=True, help="Backbone to train")
    parser.add_argument('--ensemble',type=str, choices=['bagging', 'adaboost'],required=True, help="Ensemble way to train")
    parser.add_argument('--hyper',type=int, required=True, help="Hyperparameter for train")
    args = parser.parse_args()

    print('===> Training <===')
    start = datetime.datetime.now()
    trainer = Trainer()
    trainer.set_params(args.backbone, args.ensemble, args.hyper)
    trainer.train()
    end = datetime.datetime.now()
    print('Time cost: {}'.format(end -start))
    print('===> Completed! <===')
    print('-' * 40)
