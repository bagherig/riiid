#!/usr/bin/env python
# coding: utf-8

# %% [code]
# TODO:
# Implement save parameter in train().

# %% [code]
RANDOM_SEED = 44

# %% [code]
# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load
import os
import warnings
import random
import joblib

warnings.filterwarnings('ignore')
random.seed(RANDOM_SEED)
import numpy as np  # linear algebra

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostClassifier
from sklearn import metrics
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler

from tensorflow.keras.models import load_model
from keras.callbacks import EarlyStopping
from keras.layers import Dense, Dropout, BatchNormalization
from keras.models import Sequential
from keras import backend as kb
from keras import optimizers
import tensorflow as tf
tf.random.set_seed(RANDOM_SEED)


# %% [code]
class StackingClassifier(object):
    def __init__(self, classifiers, base_classifier):
        self.classifiers = classifiers
        self.nclassifiers = len(classifiers)
        self.base_classifier = base_classifier

    def train(self, train, valid, nfolds=5):
        kfold = KFold(n_splits=nfolds, shuffle=True, random_state=RANDOM_SEED)

        base_xtrn = np.zeros(shape=(train['x'].shape[0], self.nclassifiers))
        for iclf, clf in enumerate(self.classifiers):
            print(f'\nTraining {type(clf).__name__} model.')
            print('Best hyperparameters:', clf.params)
            base_xtrn[:, iclf] = self._train_clf(clf, train, valid, kfold)
        base_xval = self._predict_clfs(valid['x'])

        print('\nTraining base classifier...')
        self.base_classifier.train(train={'x': base_xtrn, 'y': train['y']},
                                   valid={'x': base_xval, 'y': valid['y']})
        print('Validation Score:', self.base_classifier.best_score)
        print('Done.')

    def _train_clf(self, clf, train, valid, kfold):
        val_preds = np.zeros(shape=(train['x'].shape[0],))
        for ifold, (trn_idx, val_idx) in enumerate(kfold.split(train['x'])):
            print(f'--Fold {ifold + 1}...')
            xtrn = train['x'].loc[trn_idx]
            ytrn = train['y'].loc[trn_idx]
            xval = train['x'].loc[val_idx]
            yval = train['y'].loc[val_idx]

            clf.train(train={'x': xtrn, 'y': ytrn},
                      valid={'x': xval, 'y': yval},
                      verbose=10)
            clf_preds = clf.predict(xval)
            val_preds[val_idx] = clf_preds.flatten()
            # print("    	validation's auc", clf.best_score)

        print('--Training on all data...')
        clf.train(train={'x': train['x'], 'y': train['y']},
                  valid={'x': valid['x'], 'y': valid['y']},
                  verbose=10)
        return val_preds

    def predict(self, xtest):
        base_xtest = self._predict_clfs(xtest)
        return self.base_classifier.predict(base_xtest)

    def _predict_clfs(self, xtest):
        clf_preds = np.zeros(shape=(xtest.shape[0], self.nclassifiers))
        for iclf, clf in enumerate(self.classifiers):
            clf_preds[:, iclf] = clf.predict(xtest).flatten()
        return clf_preds

    def score(self, xtest, ytest):
        preds = self.predict(xtest)
        return metrics.roc_auc_score(ytest, preds)

    def save(self, path):
        if not os.path.exists(path):
            os.makedirs(path)

        for iclf, clf in enumerate(self.classifiers):
            clf.save(path=path, filename=f'{iclf}_{type(clf).__name__}')
        self.base_classifier.save(path=path, filename=f'base_{type(self.base_classifier).__name__}')


# %% [code]

class LGBMClassifier(object):
    def __init__(self, params={}, epochs=None):
        self.params = params
        self.params['feature_fraction_seed'] = RANDOM_SEED
        self.params['bagging_seed'] = RANDOM_SEED
        if epochs is not None:
            self.params['num_iterations'] = epochs
        self.log = {}
        self.clf = None
        self.best_score = 0

    def train(self, train, valid, verbose=0):
        train = lgb.Dataset(train['x'], label=train['y'], free_raw_data=True)
        valid = lgb.Dataset(valid['x'], label=valid['y'], free_raw_data=True, reference=train)
        self.log = {}
        self.clf = lgb.train(self.params,
                             train,
                             valid_sets=[valid],
                             callbacks=[lgb.record_evaluation(self.log),
                                        lgb.early_stopping(12, verbose=verbose)],
                             verbose_eval=verbose)
        self.best_score = max(self.log['valid_0'][self.params['metric']])

    def predict(self, xtest):
        return self.clf.predict(xtest)

    def save(self, path, filename):
        self.clf.save_model(path + filename + '.txt', num_iteration=self.clf.best_iteration)

    @staticmethod
    def load(modelpath):
        model = lgb.Booster(model_file=modelpath)
        clf = LGBMClassifier()
        clf.clf = model
        return clf


# %% [code]
class XGBClassifier(object):
    def __init__(self, params={}, epochs=None):
        self.params = params
        if epochs is not None:
            self.params['nrounds'] = epochs
        self.params['seed'] = RANDOM_SEED
        self.nrounds = params.pop('nrounds', 100)
        self.clf = None
        self.log = {}
        self.best_score = 0

    def train(self, train, valid, verbose=1):
        train = xgb.DMatrix(train['x'], label=train['y'])
        valid = xgb.DMatrix(valid['x'], label=valid['y'])
        self.log = {}
        self.clf = xgb.train(self.params,
                             train,
                             evals=[(valid, 'valid')],
                             num_boost_round=self.nrounds,
                             early_stopping_rounds=12,
                             verbose_eval=verbose)
        self.best_score = self.clf.best_score

    def predict(self, xtest):
        return self.clf.predict(xgb.DMatrix(xtest))

    def save(self, path, filename):
        self.clf.save_model(path + filename + '.txt')

    @staticmethod
    def load(modelpath):
        model = xgb.Booster({'nthread': 4})  # init model
        model.load_model(modelpath)  # load data
        clf = XGBClassifier()
        clf.clf = model
        return clf


# %% [code]
class CATBClassifier(object):
    def __init__(self, params={}, epochs=None):
        self.params = params
        if epochs is not None:
            self.params['iterations'] = epochs
        self.params['random_seed'] = RANDOM_SEED
        self.clf = None
        self.best_score = 0

    def train(self, train, valid, verbose=1):
        self.clf = CatBoostClassifier(verbose=verbose, **self.params)
        self.clf.fit(train['x'], train['y'],
                     eval_set=(valid['x'], valid['y']),
                     verbose=verbose,
                     use_best_model=True,
                     early_stopping_rounds=12)
        self.best_score = self.clf.get_best_score()['validation'][self.params['eval_metric']]

    def predict(self, xtest):
        return self.clf.predict_proba(xtest)[:, 1]

    def save(self, path, filename):
        self.clf.save_model(path + filename + '.txt')

    @staticmethod
    def load(modelpath):
        model = CatBoostClassifier()
        model.load_model(modelpath)
        clf = CATBClassifier()
        clf.clf = model
        return clf


# %% [code]
class SklearnClassifier(object):
    def __init__(self, **params):
        self.params = params
        self.params['random_state'] = RANDOM_SEED
        self.clf_type = self.params.pop('clf_type', None)
        self.clf = None
        self.best_score = 0

    def train(self, train, valid, verbose=0):
        self.clf = self.clf_type(**self.params, verbose=verbose)
        self.clf.fit(train['x'], train['y'])
        self.best_score = self.score(valid['x'], valid['y'])

    def predict(self, xtest):
        return self.clf.predict_proba(xtest)[:, 1]

    def score(self, xtest, ytest):
        preds = self.predict(xtest)
        return metrics.roc_auc_score(ytest, preds)

    def save(self, path, filename):
        joblib.dump(self.clf, path + filename + '.joblib')

    @staticmethod
    def load(modelpath):
        model = joblib.load(modelpath)
        clf = SklearnClassifier()
        clf.clf = model
        return clf


# %% [code]
class NeuralNetworkClassifier(object):
    def __init__(self, params={}, epochs=None):
        self.params = params
        if epochs is not None:
            self.params['epochs'] = epochs
        self.clf = None
        self.epochs = self.params.pop('epochs', 5)
        self.batch_size = self.params.pop('batch_size', 1000)
        self.metric = self.params.pop('metric_name', None)
        self.scaler = self.params.pop('scaler', MinMaxScaler(feature_range=(0, 1)))
        self.log = {}
        self.best_score = 0

    def train(self, train, valid, verbose=0):
        verbose = min(1, verbose)
        train['x'] = self.scaler.fit_transform(train['x'])
        valid['x'] = self.scaler.transform(valid['x'])
        self.clf = self._create_ann(in_shape=train['x'].shape[1:],
                                    out_shape=1,
                                    **self.params)
        callback_es = EarlyStopping(monitor=f'val_{self.metric}',
                                    patience=10, mode='max', verbose=verbose)
        self.log = self.clf.fit(train['x'], train['y'],
                                epochs=self.epochs,
                                batch_size=self.batch_size,
                                validation_data=(valid['x'], valid['y']),
                                callbacks=[callback_es],
                                shuffle=True,
                                verbose=verbose)
        self.best_score = max(self.log.history[f'val_{self.metric}'])

    def predict(self, xtest):
        xtest = self.scaler.transform(xtest)
        return self.clf.predict(xtest)

    def save(self, path, filename):
        self.clf.save(path + filename + '.h5')
        joblib.dump(self.scaler, path + filename + '_scaler.pkl')

    @staticmethod
    def load(modelpath, scalerpath):
        model = load_model(modelpath)
        clf = NeuralNetworkClassifier()
        clf.clf = model
        clf.scaler = joblib.load(scalerpath)
        return clf

    @staticmethod
    def _create_ann(in_shape,
                    out_shape,
                    loss,
                    metric,
                    n_layers=1,
                    last_units=64,
                    units_decay=1,
                    hidden_sizes=None,
                    learning_rate=0.001,
                    lr_decay=0.0,
                    dropout_rate=0.1,
                    norm=0,
                    min_units=10,
                    activation='relu'):
        """
        Creates a multilayer perceptron model.

        :param in_shape: The shape of the input data.
        :param out_shape: The shape of the model output.
        :param n_layers: The number of hidden dense layers.
        :param last_units: The number of units for the last hidden dense layer.
        :param units_decay: The decay rate of the number of units going from last hidden layer to the first.
        :param hidden_sizes: A list containing the number of nodes for each hidden dense layer. This is meant to be used as 
            an alternative to n_layers, n_units, and units_scale to specify the number of hidden layers and the number of 
            nodes per hidden layer.
        :param learning_rate: The learning rate of the model.
        :param lr_decay: The learning rate decay of the model.
        :param dropout_rate: The dropout rate for the dropout layers.
        :param norm: Whether to use a batch normalization layer after each dense layer or not.

        :return An MLP model.
        """
        kb.clear_session()
        tf.compat.v1.reset_default_graph()

        n_layers = int(np.floor(n_layers))
        last_units = int(last_units)
        norm = int(round(norm))
        model = Sequential()

        # Adding the input and hidden layers.
        if hidden_sizes is not None:
            for i_layer, i_units in enumerate(hidden_sizes):
                if i_layer == 0:
                    model.add(Dense(i_units, activation=activation,
                                    input_shape=in_shape))
                else:
                    model.add(Dense(i_units, activation=activation))
                if norm:
                    model.add(BatchNormalization())
                if dropout_rate:
                    model.add(Dropout(dropout_rate))
        else:
            for i_layer in range(n_layers):
                if i_layer == 0:
                    model.add(Dense(max(min_units, int(last_units * (units_decay ** (n_layers - i_layer - 1)))),
                                    activation=activation,
                                    input_shape=in_shape))
                else:
                    model.add(Dense(max(min_units, int(last_units * (units_decay ** (n_layers - i_layer - 1)))),
                                    activation=activation))
                if norm:
                    model.add(BatchNormalization())
                if dropout_rate:
                    model.add(Dropout(dropout_rate))

                    # Adding the output layer
        model.add(Dense(units=out_shape, activation='sigmoid'))

        model.compile(optimizer=optimizers.Adam(lr=learning_rate, decay=lr_decay),
                      loss=loss, metrics=metric)

        return model

