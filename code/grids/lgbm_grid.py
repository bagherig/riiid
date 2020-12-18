fixed_params = {
    'num_iterations': 100,
    'metric': 'auc',
    'objective': 'binary',
    'boosting_type': 'gbdt',
    'is_unbalance': True,
    'force_row_wise': True,
    'num_threads': 6,
    'verbose': 0
}

opt_grid = {
    'learning_rate': (0.001, 0.2),
    'feature_fraction': (0.5, 0.9),
    'lambda_l2': (0, 5),
    'num_leaves': (50, 1500),
    'min_data_in_leaf': (10, 2000)
}

opt_dtypes = {
    'learning_rate': float,
    'feature_fraction': float,
    'lambda_l2': float,
    'num_leaves': int,
    'min_data_in_leaf': int
}

probes = [
    {"feature_fraction": 0.689624417022823,
     "lambda_l2": 4.450308176307347,
     "learning_rate": 0.12494856029217047,
     "min_data_in_leaf": 1513,
     "num_leaves": 741}
]