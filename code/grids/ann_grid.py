import tensorflow as tf
from sklearn.preprocessing import MinMaxScaler

fixed_params = {
    'epochs': 5,
    'batch_size': 10_000,
    'norm': 1,
    'loss': 'binary_crossentropy',
    'metric': tf.keras.metrics.AUC(),
    'metric_name': 'auc',
    'activation': 'relu',
    'scaler': MinMaxScaler(feature_range=(0, 1)),
    'min_units': 64
}

opt_grid = {
    'n_layers': (3, 5.99),
    'last_units': (64, 512),
    'units_decay': (0, 0.8),
    'learning_rate': (1e-4, 1e-1),
    'lr_decay': (0, 1e-2),
    'dropout_rate': (0, 0.4)
}

opt_dtypes = {
    'n_layers': int,
    'last_units': int,
    'units_decay': float,
    'learning_rate': float,
    'lr_decay': float,
    'dropout_rate': float
}

probes = [
    {"dropout_rate": 0.03770705373918477,
     "last_units": 366.38259494642006,
     "learning_rate": 0.0069023355794437915,
     "lr_decay": 0.00035413861814238015,
     "n_layers": 3.330988607380866,
     "units_decay": 0.6404255252271216}
]