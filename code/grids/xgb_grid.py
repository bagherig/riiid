fixed_params = {
    'nrounds': 50,
    'eval_metric': 'auc',
    'objective': 'binary:logistic',
    'tree_method': 'exact',
    'num_parallel_tree': 1,
    'verbosity': 0
}

opt_grid = {
    'learning_rate': (0.001, 0.2),
    'colsample_bytree': (0.5, 0.9),
    'subsample': (0.5, 1),
    'lambda': (1, 100),
    'max_depth': (5, 7.99),
    'min_child_weight': (1, 2000)
}

opt_dtypes = {
    'learning_rate': float,
    'colsample_bytree': float,
    'subsample': float,
    'lambda': float,
    'max_depth': int,
    'min_child_weight': int
}

probes = [
    {"learning_rate": 0.12494856029217047,
     "colsample_bytree": 0.689624417022823,
     "subsample": 1,
     "lambda": 4.450308176307347,
     "max_depth": 6,
     "min_child_weight": 1513},
    {"learning_rate": 0.18,
     "colsample_bytree": 0.5014,
     "subsample": 0.7202,
     "lambda": 1.654,
     "max_depth": 6,
     "min_child_weight": 348}
]