#!/usr/bin/env python
# coding: utf-8

# %% [code]
import gc
import time
import os
import json
import pickle
import importlib
from shutil import copyfile
import numpy as np
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import classifiers as sc

from sklearn.linear_model import LogisticRegression

from functools import partial
from bayes_opt import BayesianOptimization, SequentialDomainReductionTransformer
from bayes_opt.logger import JSONLogger, ScreenLogger
from bayes_opt.event import Events
from bayes_opt.util import load_logs

# %% [code]
PARENT_DIR = os.path.dirname(__file__) + '/../'
PARQUETS_DIR = PARENT_DIR + 'data/parquets/'
MODELS_DIR = PARENT_DIR + 'models/'

OUT_DIR = PARENT_DIR + 'temp/'


# %% [code]
class Timer(object):
    def __init__(self, engine):
        engine.subscribe(Events.OPTIMIZATION_START, self)
        engine.subscribe(Events.OPTIMIZATION_STEP, self)
        self.init_time = None

    def update(self, event, instance):
        if event == Events.OPTIMIZATION_START:
            self.init_time = time.time()
        else:
            time_taken = time.time() - self.init_time
            print(f"Time: {time.strftime('%H:%M:%S', time.localtime(time.time()))} | "
                  f"Taken: {time.strftime('%H:%M:%S', time.gmtime(time_taken))}", end='\r')
            self.init_time = time.time()


# %% [code]
def prepare_params(opt_params, opt_dtypes):
    for param in opt_params:
        opt_params[param] = opt_dtypes[param](opt_params[param])


def scorer(clf, train, valid, fixed_params, opt_dtypes, **opt_params):
    prepare_params(opt_params, opt_dtypes)
    opt_params.update(fixed_params)
    model = clf(opt_params)
    model.train(train, valid, verbose=0)

    return model.best_score


def get_scorer(clf, train, valid, fixed_params, opt_dtypes):
    return partial(scorer, clf, train, valid, fixed_params, opt_dtypes)


def get_optimizer(scorer_func, params_grid, filename, cont_opt=False, probes=[], use_transformer=False):
    transformer = SequentialDomainReductionTransformer(gamma_osc=0.8, eta=0.95)
    optimizer = BayesianOptimization(f=scorer_func,
                                     pbounds=params_grid,
                                     verbose=2,
                                     bounds_transformer=transformer if use_transformer else None)

    scr_logger = ScreenLogger()
    scr_logger._default_cell_size = 10
    optimizer.subscribe(Events.OPTIMIZATION_START, scr_logger)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, scr_logger)
    optimizer.subscribe(Events.OPTIMIZATION_END, scr_logger)

    if cont_opt:
        if os.path.exists(filename):
            optimizer.dispatch(Events.OPTIMIZATION_START)
            optimizer.unsubscribe(Events.OPTIMIZATION_START, scr_logger)
            load_logs(optimizer, logs=[filename])
        else:
            print('File not found:', filename)
    else:
        for ps in probes:
            optimizer.probe(lazy=True, params=ps)

    f_logger = JSONLogger(path=filename, reset=False if cont_opt else True)
    optimizer.subscribe(Events.OPTIMIZATION_STEP, f_logger)
    Timer(optimizer)

    return optimizer


# %% [code]
def get_data():
    data = pd.read_parquet(PARQUETS_DIR + 'train_df.parquet', columns=FEATURES+[TARGET])
    train, valid = split_train_valid(data, val_size=1_000_000).values()

    return train, valid


def split_train_valid(dt, val_size):
    val = dt.iloc[-val_size:]
    trn = dt.iloc[:-val_size]
    xtrn, ytrn = trn.drop(columns=[TARGET]), trn[TARGET]
    xval, yval = val.drop(columns=[TARGET]), val[TARGET]

    return {'trn': {'x': xtrn, 'y': ytrn},
            'val': {'x': xval, 'y': yval}}


def optimize(clf, label, init_points=10, cont_opt=False, version='v1', **kwargs):
    opt_filename = f'{label}_opt.json'
    clf_dicts = importlib.import_module(f'grids.{label}_grid')
    if cont_opt and not os.path.exists(OUT_DIR + opt_filename):
        copyfile(MODELS_DIR + f'{label}/{version}/' + opt_filename, OUT_DIR + opt_filename)

    train, valid = get_data()
    scorer_func = get_scorer(clf, train, valid, clf_dicts.fixed_params, clf_dicts.opt_dtypes)
    optimizer = get_optimizer(scorer_func,
                              params_grid=clf_dicts.opt_grid,
                              probes=clf_dicts.probes,
                              cont_opt=cont_opt,
                              filename=OUT_DIR + opt_filename,
                              **kwargs)
    optimizer.maximize(init_points=init_points, n_iter=200, alpha=1e-6)


# %% [code]
def get_best_params(label, version='v1'):
    clf_dicts = importlib.import_module(f'grids.{label}_grid')
    scores, params = [], []
    with open(MODELS_DIR + f'{label}/{version}/{label}_opt.json', 'r') as f:
        opt_hist = f.readlines()
    for line in opt_hist:
        hist = json.loads(line)
        scores.append(hist['target'])
        params.append(hist['params'])

    best_iter = np.argmax(scores)
    best_params = params[best_iter]
    prepare_params(best_params, clf_dicts.opt_dtypes)
    best_params.update(clf_dicts.fixed_params)

    return best_params


# %% [code]
TARGET = 'answered_correctly'
FEATURES = [
    #     'user_id',
    #     'content_id',
    'prior_question_elapsed_time',
    #     'prior_question_had_explanation',
    'part',
    #     'tag1',
    #     'tag2',
    #     'tag3',
    #     'tag4',
    #     'tag5',
    #     'tag6',
    'user_id_count',
    'user_id_wmean',
    'user_id_attempts',
    'content_id_count',
    'content_id_mean',
    'tag_count_0',
    'tag_count_1',
    'tag_count_2',
    'tag_count_3',
    #     'tag_count_4',
    #     'tag_count_5',
    'tag_mean_0',
    'tag_mean_1',
    'tag_mean_2',
    'tag_mean_3',
    #     'tag_mean_4',
    #     'tag_mean_5',
    'user_id_tag_count_0',
    'user_id_tag_count_1',
    'user_id_tag_count_2',
    'user_id_tag_count_3',
    #     'user_id_tag_count_4',
    #     'user_id_tag_count_5',
    'user_id_tag_mean_0',
    'user_id_tag_mean_1',
    'user_id_tag_mean_2',
    'user_id_tag_mean_3',
    #     'user_id_tag_mean_4',
    #     'user_id_tag_mean_5',
    'user_content_hmean',
    #     'tags_hmean',
    'tags_whmean',
    #     'user_tags_hmean'
    'user_tags_whmean'
]

CAT_FEATURES = [
    'part'
]

# %% [markdown]
# # Hyperparameter Optimization


# %% [code]
clf = sc.NeuralNetworkClassifier
label = 'ann'
if 0:
    optimize(clf=clf,
             label=label,
             init_points=0,
             cont_opt=True,
             use_transformer=False)


# %% [code]
lgbm_model = sc.LGBMClassifier(get_best_params('lgbm', version='v1'), epochs=1000)
xgb_model = sc.XGBClassifier(get_best_params('xgb', version='v1'), epochs=1000)
catb_model = sc.CATBClassifier(get_best_params('catb', version='v1'), epochs=1000)
ann_model = sc.NeuralNetworkClassifier(get_best_params('ann', version='v1'), epochs=1000)

base_model = sc.SklearnClassifier(clf_type=LogisticRegression)

model = sc.StackingClassifier(classifiers=[lgbm_model, xgb_model, catb_model, ann_model],
                              base_classifier=base_model)

train_data, valid_data = get_data()
model.train(train_data, valid_data, nfolds=4)
model_dir = OUT_DIR + f'model_{int(time.time())}/'
print(model.score(valid_data['x'], valid_data['y']))
model.save(model_dir)

# %% [code]
# lgbm_model = sc.LGBMClassifier.load(model_dir + '0_LGBMClassifier.txt')
# xgb_model = sc.XGBClassifier.load(model_dir + '1_XGBClassifier.txt')
# catb_model = sc.CATBClassifier.load(model_dir + '2_CATBClassifier.txt')
# ann_model = sc.NeuralNetworkClassifier.load(modelpath=model_dir + '3_NeuralNetworkClassifier.h5',
#                                             scalerpath=model_dir + '3_NeuralNetworkClassifier_scaler.pkl')
#
# base_model = sc.SklearnClassifier.load(model_dir + 'base_SklearnClassifier.joblib')
#
# model2 = sc.StackingClassifier(classifiers=[lgbm_model, xgb_model, catb_model, ann_model],
#                               base_classifier=base_model)
# print(model2.score(valid_data['x'], valid_data['y']))
