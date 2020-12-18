fixed_params = {
    'iterations': 50,
    'bootstrap_type': 'Bernoulli',
    'eval_metric': 'AUC',
    'grow_policy': 'Lossguide',
    'allow_writing_files': False,
    'od_type': 'Iter',
    'auto_class_weights': 'Balanced',
    'use_best_model': True
}

opt_grid = {
    'learning_rate': (0.001, 0.2),
    'depth': (8, 16.99),
    'l2_leaf_reg': (1, 100),
    'colsample_bylevel': (0.5, 1),
    'subsample': (0.6, 1),
    'min_data_in_leaf': (1, 2000),
    'max_leaves': (50, 2000)
}

opt_dtypes = {
    'learning_rate': float,
    'depth': int,
    'l2_leaf_reg': float,
    'colsample_bylevel': float,
    'subsample': float,
    'min_data_in_leaf': int,
    'max_leaves': int
}

probes = [
    {"learning_rate": 0.12494856029217047,
     "colsample_bylevel": 0.689624417022823,
     "subsample": 0.9,
     "l2_leaf_reg": 4.450308176307347,
     "depth": 6,
     "min_data_in_leaf": 1513,
     "max_leaves": 741},
]