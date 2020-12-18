from sklearn.ensemble import RandomForestClassifier

fixed_params = {
    'n_jobs': 6,
    'clf_type': RandomForestClassifier
}

opt_grid = {
    'max_depth': (6, 10.99),
    'n_estimators': (10, 128),
    'max_features': (0.5, 1),
    'max_samples': (0.5, 1),
    'min_samples_leaf': (1, 2000),
    'min_samples_split': (1, 2000),
}

opt_dtypes = {
    'max_depth': int,
    'n_estimators': int,
    'max_features': float,
    'max_samples': float,
    'min_samples_leaf': int,
    'min_samples_split': int,
}

probes = [
    {'max_depth': 6,
     'n_estimators': 10,
     'max_features': 0.9,
     'max_samples': 0.9,
     'min_samples_leaf': 1513,
     'min_samples_split': 1513}
]