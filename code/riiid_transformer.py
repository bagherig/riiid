# -*- coding: utf-8 -*-
"""riiid-transformer.ipynb"""

"""# Imports"""

# import riiideducation

import os
import gc
import sys
import math
import random
import psutil
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

warnings.filterwarnings('ignore')
gc.enable()

def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
RANDOM_SEED = 1
seed_everything(RANDOM_SEED)
print(torch.__version__)


from psutil import virtual_memory

ram_gb = virtual_memory().total / 1e9
print('Your runtime has {:.1f} gigabytes of available RAM'.format(ram_gb))

if ram_gb < 20:
    print(
        'To enable a high-RAM runtime, select the Runtime > "Change runtime type"')
    print(
        'menu, and then select High-RAM in the Runtime shape dropdown. Then, ')
    print('re-execute this cell.')
else:
    print('You are using a high-RAM runtime!')

"""# Global Variables"""

# %% [code]

COMPILE = False
VERSION = 'v2'
TPU = False

PARENT_DIR = os.path.dirname(__file__) + '/../'
DATA_DIR = PARENT_DIR + '/data/riiid-test-answer-prediction/'
PARQUETS_DIR = PARENT_DIR + 'data/parquets/transformer/'
MODELS_DIR = PARENT_DIR + 'models/'

OUT_DIR = PARENT_DIR + 'temp/'


# %% [code]

TARGET = 'answered_correctly'
KEY_FEATURE = 'user_id'
FEATURES = [
    'content_id',
    'timestamp',
    'prior_question_elapsed_time',
    'prior_question_had_explanation',
    # 'task_container_id',
    'part',
    'tag1',
    'tag2',
    # 'tag3',
    #     'tag4',
    #     'tag5',
    #     'tag6',
    # 'content_mean',
    # 'usercontent_mean',
]

DTYPES = {
    'content_id': int,
    'timestamp': int,
    'prior_question_elapsed_time': int,
    'prior_question_had_explanation': int,
    # 'task_container_id': int,
    'part': int,
    'tag1': int,
    'tag2': int,
    # 'tag3': int,
    #     'tag4': int,
    #     'tag5': int,
    #     'tag6': int,
    # 'content_mean': int,
    # 'usercontent_mean': int,
}

# For Inference
ADDED_FEATURES = [
    'part',
    'tag1',
    'tag2',
    # 'tag3',
    #     'tag4',
    #     'tag5',
    #     'tag6',
    # 'content_mean',
    # 'usercontent_mean',
]

DEFAULT_VALUES = {
    'prior_question_had_explanation': False,
    'prior_question_elapsed_time': 26000,
    'timestamp': 0,
    'part': 0,
    'tag1': -1,
    'tag2': -1,
    # 'tag3': -1,
    # 'tag4': -1,
    # 'tag5': -1,
    # 'tag6': -1,
}

SUBMISSION_COLUMNS = [
    'row_id',
    TARGET
]


# %% [code]
def prepare_data():
    print("Reading datasets...")
    dt = pd.read_parquet(PARQUETS_DIR + 'train.parquet')
    questions = pd.read_parquet(PARQUETS_DIR + 'questions.parquet')

    print("Merging datasets...")
    dt = dt[(dt['content_type_id'] == 0) & (dt['answered_correctly'] != -1)] \
        .drop(columns=['content_type_id'])
    questions.index.name = 'content_id'
    questions = questions.reset_index()
    dt = dt.merge(questions, on='content_id', how='left')
    dt = dt.sort_values(['timestamp'], ascending=True).reset_index(drop=True)
    print("Getting content statistics...")
    q_means = dt[['content_id', TARGET]]\
        .groupby('content_id')[TARGET]\
        .agg(['cumsum', 'cumcount']) \
        .shift(fill_value=0)
    dt['content_mean'] = q_means['cumsum'] / (q_means['cumcount'] + 1)

    print("Getting user-content statistics...")
    qu_means = dt[['content_id', KEY_FEATURE, TARGET]]\
        .groupby(['content_id', KEY_FEATURE])[TARGET] \
        .agg(['cumsum', 'cumcount']) \
        .shift(fill_value=0)
    dt['usercontent_mean'] = qu_means['cumsum'] / (qu_means['cumcount'] + 1)

    print('Splitting train, valid, test...')
    trn, tst = split_train_valid(dt, 0.025)

    print('Writing datasets to .parquet files...')
    dt.to_parquet(PARQUETS_DIR + 'train_merged.parquet')
    trn.to_parquet(PARQUETS_DIR + 'train_transformer.parquet')
    tst.to_parquet(PARQUETS_DIR + 'test_transformer.parquet')


def load_data(filename):
    return pd.read_parquet(PARQUETS_DIR + filename,
                           columns=[KEY_FEATURE] + FEATURES + [TARGET])


def preprocess(dt: pd.DataFrame):
    dt["content_id"] += 1
    if 'prior_question_elapsed_time' in dt.columns:
        dt["prior_question_elapsed_time"] = \
            dt["prior_question_elapsed_time"].fillna(DEFAULT_VALUES['prior_question_elapsed_time'])
        dt["prior_question_elapsed_time"] = \
            np.ceil(dt["prior_question_elapsed_time"] / 1000).astype(np.int32)
    if 'prior_question_had_explanation' in dt.columns:
        dt['prior_question_had_explanation'].fillna(DEFAULT_VALUES['prior_question_had_explanation'], inplace=True)
    for col in ["task_container_id", "tag1", "tag2", "tag3", "tag4", "tag5", "tag6"]:
        if col in dt.columns:
            dt[col] += 1
    dt['position'] = dt[[KEY_FEATURE, TARGET]].groupby(KEY_FEATURE)[TARGET].cumcount() + 1

    return dt

def preprocess_timestamps(dt: pd.DataFrame):
    if 'timestamp' in dt.columns:
        timestamps_raw = dt['timestamp'].values
        dt['timestamp'] = dt[[KEY_FEATURE, 'timestamp']]\
            .groupby(KEY_FEATURE)['timestamp'].diff().fillna(DEFAULT_VALUES['timestamp'])
        dt['timestamp'] = np.ceil(dt['timestamp'] / (1000)).astype(np.int32)
        dt['timestamp_raw'] = timestamps_raw
    return dt

def preprocess_test(dt: np.ndarray, all_cols):
    dt[:, all_cols["content_id"]] += 1
    if 'prior_question_had_explanation' in all_cols:
        time_nans = pd.isnull(dt[:, all_cols["prior_question_elapsed_time"]])
        dt[time_nans, all_cols["prior_question_elapsed_time"]] = \
            DEFAULT_VALUES['prior_question_elapsed_time']
        dt[:, all_cols["prior_question_elapsed_time"]] = \
            np.ceil(dt[:, all_cols["prior_question_elapsed_time"]] / 1000).astype(np.int32)
    if 'prior_question_had_explanation' in all_cols:
        explanation_nans = pd.isnull(dt[:, all_cols["prior_question_had_explanation"]])
        dt[explanation_nans, all_cols["prior_question_had_explanation"]] = \
            DEFAULT_VALUES['prior_question_had_explanation']
    for col in ["task_container_id", "tag1", "tag2", "tag3", "tag4", "tag5", "tag6"]:
        if col in all_cols:
            dt[:, all_cols[col]] += 1
    return dt


def split_train_valid(dt, val_fraction):
    if val_fraction == 1:
        return None, dt
    val_size = 0
    trn_size = 0
    val_uids = []
    val_fraction = val_fraction / (1 - val_fraction)
    n_samples_per_user = dt.groupby(KEY_FEATURE)[
        TARGET].count().sort_values().reset_index().values.tolist()
    while n_samples_per_user:
        uid, nsamples = n_samples_per_user.pop()
        if trn_size * val_fraction >= val_size:
            val_uids.append(uid)
            val_size += nsamples
        else:
            trn_size += nsamples

    val = dt[dt[KEY_FEATURE].isin(val_uids)]
    trn = dt.drop(val.index)
    # val_size = int(dt.shape[0] * val_fraction)
    # trn = dt[:-val_size]
    # val = dt[-val_size:]
    return trn, val


def pad_batch(x, window_size, pad_value=0):
    pad_dims = (window_size - x.shape[0], 0)
    shape = (pad_dims,) + tuple(
        (0, 0) for i in range(len(x.shape) - 1))
    return np.pad(x, shape, constant_values=pad_value)


def create_scheduler(estimator, optim, warmup_steps=10, last_epoch=-1):
    sched = None
    if SCHED_TYPE == 'lambda':
        lr_lambda = lambda epoch: ((estimator.d_model ** -0.5) *
                                   min(((epoch + 1) ** -0.5),
                                       (epoch + 1) * (warmup_steps ** -1.5)))
        sched = torch.optim.lr_scheduler.LambdaLR(optim,
                                                  lr_lambda=lr_lambda,
                                                  last_epoch=last_epoch)
    elif SCHED_TYPE == 'plateau':
        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(optim, 'max',
                                                           patience=0,
                                                           factor=0.2,
                                                           min_lr=1e-7)
    return sched


def create_optimizer(estimator, lr):
    return torch.optim.Adam(estimator.parameters(), lr=lr)


def create_model(model_type, **params):
    return model_type(**params).to(DEVICE)


def save_model(estimator, optim, sched=None, val_score: float = 0):
    checkpoint = {'model_params': estimator.params,
                  'model_state_dict': estimator.state_dict(),
                  'optimizer_state_dict': optim.state_dict(),
                  'learning_rate': optim.param_groups[0]['lr'],
                  'val_score': val_score}
    if sched is not None:
        checkpoint = {**checkpoint,
                      'sheduler_state_dict': sched.state_dict(),
                      'epoch': sched.last_epoch}
    torch.save(checkpoint, OUT_DIR + MODEL_FILENAME)


def load_model(model_type, for_training=False, warmup_steps=10):
    if os.path.exists(OUT_DIR + MODEL_FILENAME):
        filepath = OUT_DIR + MODEL_FILENAME
    else:
        filepath = MODELS_DIR + f'transformer/{VERSION}/{MODEL_FILENAME}'

    print(f'Loading model from {filepath}')
    checkpoint = torch.load(filepath, map_location=DEVICE)

    estimator = create_model(model_type, **checkpoint['model_params'])
    estimator.load_state_dict(checkpoint['model_state_dict'])

    optim = None
    sched = None
    if for_training:
        optim = create_optimizer(estimator, lr=checkpoint['learning_rate'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'sheduler_state_dict' in checkpoint:
            sched = create_scheduler(estimator, optim, warmup_steps,
                                     last_epoch=checkpoint['epoch'])
            sched.load_state_dict(checkpoint['sheduler_state_dict'])

    return estimator, optim, sched, checkpoint['val_score']

# %% [code]

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
                -math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, pos):
        return self.pe[pos, :]


class FFN(nn.Module):
    def __init__(self, d_model, dim_feedforward, dropout, activation):
        super(FFN, self).__init__()
        self.lr1 = nn.Linear(d_model, dim_feedforward)
        self.activation = self._get_activation_fn(activation)
        self.lr2 = nn.Linear(dim_feedforward, d_model)
        self.dropout = nn.Dropout(dropout)

    def _get_activation_fn(self, activation):
        if activation == "relu":
            return nn.ReLU()
        elif activation == "gelu":
            return nn.GELU()
        elif activation == "tanh":
            return nn.Tanh()
        elif activation == "sigmoid":
            return nn.Sigmoid()
        raise RuntimeError("activation should be relu/gelu/tanh/sigmoid, not {}"\
                           .format(activation))

    def forward(self, x):
        x = self.lr1(x)
        x = self.activation(x)
        x = self.lr2(x)
        return self.dropout(x)


"""# TransformerEncoder Model2"""


class TransformerEncoderModel2(nn.Module):
    def __init__(self, **params):
        print(params)
        super().__init__()
        self.params = params.copy()
        self.d_model = params['d_model']
        self.seq_len = params.pop('seq_len') - 1
        n_enc_layers = params.pop('num_encoder_layers')
        n_dec_layers = params.pop('num_decoder_layers')
        dropout_rate = params['dropout']

        # self.norm = nn.LayerNorm(self.d_model, elementwise_affine=False)
        # self.ts_norm = nn.LayerNorm(self.seq_len)
        # self.et_norm = nn.LayerNorm(self.seq_len)
        self.enc_norm = nn.LayerNorm(self.d_model)
        self.dec_norm = nn.LayerNorm(self.d_model)

        encoder_layer = nn.TransformerEncoderLayer(**params)
        self.encoder1 = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                              num_layers=n_enc_layers,
                                              norm=self.enc_norm)
        self.encoder2 = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                              num_layers=n_enc_layers,
                                              norm=self.enc_norm)
        self.encoder3 = nn.TransformerEncoder(encoder_layer=encoder_layer,
                                              num_layers=n_enc_layers,
                                              norm=self.enc_norm)
        decoder_layer = nn.TransformerDecoderLayer(**params)
        self.decoder = nn.TransformerDecoder(decoder_layer=decoder_layer,
                                             num_layers=n_dec_layers,
                                             norm=self.dec_norm)
        self.max_exercise = 13523
        self.max_part = 7
        self.max_explanation = 1
        self.max_tag = 188
        self.max_target = 2
        self.max_time = 300  # seconds
        self.max_timestamp = 86400 #1440  # minutes
        self.max_position = 10000

        self.position_embeddings = PositionalEncoding(self.d_model,
                                                      self.max_position + 1)
        self.exercise_embeddings = \
            nn.Embedding(num_embeddings=self.max_exercise + 1,
                         embedding_dim=self.d_model)  # exercise_id
        self.part_embeddings = \
            nn.Embedding(num_embeddings=self.max_part + 1,
                         embedding_dim=self.d_model)
        self.explanation_embeddings = \
            nn.Embedding(num_embeddings=self.max_explanation + 1,
                         embedding_dim=self.d_model)
        self.timestamp_embeddings = (
            # nn.Linear(1, self.d_model, bias=False))
            nn.Embedding(num_embeddings=self.max_timestamp + 1,
                         embedding_dim=self.d_model))
        self.elapsed_time_embeddings = (
            # nn.Linear(1, self.d_model, bias=False))
            nn.Embedding(num_embeddings=self.max_time + 1,
                         embedding_dim=self.d_model))
        self.tag1_embeddings = \
            nn.Embedding(num_embeddings=self.max_tag + 1,
                         embedding_dim=self.d_model)
        self.tag2_embeddings = \
            nn.Embedding(num_embeddings=self.max_tag + 1,
                         embedding_dim=self.d_model)
        self.target_embeddings = \
            nn.Embedding(num_embeddings=self.max_target + 1,
                         embedding_dim=self.d_model)

        self.linear = nn.Linear(self.d_model, 1)
        self.sigmoid = nn.Sigmoid()
        self.device = DEVICE

        self.future_mask = self.generate_square_subsequent_mask(
            self.seq_len).to(self.device)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    def init_weights(self):
        initrange = 0.1
        # init embeddings
        self.exercise_embeddings.weight.data.uniform_(-initrange, initrange)
        self.part_embeddings.weight.data.uniform_(-initrange, initrange)
        self.explanation_embeddings.weight.data.uniform_(-initrange, initrange)
        self.elapsed_time_embeddings.weight.data.uniform_(-initrange,
                                                          initrange)
        self.timestamp_embeddings.weight.data.uniform_(-initrange, initrange)
        self.tag1_embeddings.weight.data.uniform_(-initrange, initrange)
        self.tag2_embeddings.weight.data.uniform_(-initrange, initrange)
        self.target_embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs: dict):
        """
        S is the sequence length, N the batch size and E the Embedding Dimension (number of features).
        src: (S, N, E)
        src_mask: (S, S)
        src_key_padding_mask: (N, S)
        padding mask is (N, S) with boolean True/False.
        SRC_MASK is (S, S) with float(’-inf’) and float(0.0).
        """
        (content_ids,
         timestamps,
         elapsed_times,
         explanations,
         parts,
         tags1, tags2,
         targets,
         positions) = inputs.values()  # (N, S)

        content_ids[content_ids > self.max_exercise] = 0
        timestamps[timestamps > self.max_timestamp] = self.max_timestamp
#         timestamps[timestamps <= 30] = 1
#         timestamps[(timestamps > 30) & (timestamps <= 60)] = 2
#         timestamps[(timestamps > 120) & (timestamps <= 360)] = 3
#         timestamps[timestamps > 360] = 4

        elapsed_times[elapsed_times > self.max_time] = self.max_time
        parts[parts > self.max_part] = 0
        positions[positions[:, -1] > self.max_position] = torch.arange(
            self.max_position - self.seq_len + 1, self.max_position + 1,
            device=self.device)

        # elapsed_times = self.et_norm(elapsed_times).view(-1,1)
        # timestamps = self.ts_norm(timestamps).view(-1,1)

        embedded_inp1 = (self.exercise_embeddings(content_ids)
                         + self.part_embeddings(parts)
                         + self.explanation_embeddings(explanations)
                         + self.timestamp_embeddings(timestamps)
                         + self.target_embeddings(targets)
                         ).transpose(0, 1) * np.sqrt(self.d_model)
        embedded_inp2 = (self.exercise_embeddings(content_ids)
                         + self.tag1_embeddings(tags1)
                         + self.tag2_embeddings(tags2) * 0.8
                         + self.target_embeddings(targets)
                         ).transpose(0, 1) * np.sqrt(self.d_model)
        embedded_inp3 = (self.elapsed_time_embeddings(elapsed_times)#.view(-1, self.seq_len, self.d_model)
                         + self.timestamp_embeddings(timestamps)#.view(-1, self.seq_len, self.d_model)
                         + self.target_embeddings(targets)
                         ).transpose(0, 1) * np.sqrt(self.d_model)
        # (S, N, E)
        positions[:] = torch.arange(0, self.seq_len)
        embedded_pos = self.position_embeddings(positions.transpose(0, 1))
        embedded_inp1 += embedded_pos
        embedded_inp2 += embedded_pos
        embedded_inp3 += embedded_pos

        output1 = self.encoder1(src=self.enc_norm(embedded_inp1),
                                mask=self.future_mask)  # (S, N, E)
        output2 = self.encoder2(src=self.enc_norm(embedded_inp2),
                                mask=self.future_mask)  # (S, N, E)
        output3 = self.encoder3(src=self.enc_norm(embedded_inp3),
                                mask=self.future_mask)  # (S, N, E)
        output = (self.target_embeddings(targets).transpose(0, 1)
                  + output1 + output2 + output3) * np.sqrt(self.d_model)
        tgt = (self.exercise_embeddings(content_ids)
               + self.target_embeddings(targets)
               ).transpose(0, 1) * np.sqrt(self.d_model)
        output = self.decoder(tgt=self.dec_norm(tgt + embedded_pos),
                              memory=self.dec_norm(output + embedded_pos),
                              tgt_mask=self.future_mask,
                              memory_mask=self.future_mask)  # (S, N, E)
        output = self.linear(output.transpose(1, 0)).squeeze(-1)
        output = self.sigmoid(output)

        return output


"""# Training Functions"""


# %% code[]

class Riiid(torch.utils.data.Dataset):
    def __init__(self, dt, seq_len, pad_value=0):
        super().__init__()
        self.seq_len = seq_len
        self.min_seq_len = 20
        self.pad_value = pad_value
        self.dcols = {col: i for i, col in
                      enumerate(FEATURES + [TARGET])}
        groups = dt.groupby(KEY_FEATURE).apply(
            lambda r: r[FEATURES + [TARGET, 'position']].values).values
        del dt
        self.samples = {}
        self.uids = []
        for idx in range(groups.shape[0]):
            udata = groups[idx]
            udata_len = udata.shape[0]
            if udata_len >= self.min_seq_len:
                if udata_len > self.seq_len:
                    last_pos = udata_len // self.seq_len
                    for seq in range(last_pos):
                        index = f"{idx}_{seq}"
                        self.uids.append(index)
                        start = seq * self.seq_len
                        end = (seq + 1) * self.seq_len
                        self.samples[index] = udata[-end:-start]
                    if udata_len - end >= self.min_seq_len:
                        index = f'{idx}_{last_pos + 1}'
                        self.uids.append(index)
                        self.samples[index] = udata[:-end]
                else:
                    index = f'{idx}'
                    self.uids.append(index)
                    self.samples[index] = udata
        del groups
        gc.collect()

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        uid = self.uids[idx]
        inputs = self.samples[uid]
        targets = inputs[:, self.dcols[TARGET]] + 1
        inputs = pad_batch(inputs, self.seq_len, self.pad_value)
        targets = pad_batch(targets, self.seq_len, self.pad_value)
        inputs = np.c_[inputs[1:], targets[:-1]]
        return inputs


def collate_fn(batch):
    return np.array(batch).transpose(2, 0, 1)


def train_epoch(estimator, trn_iterator, optim, criterion,
                device="cpu"):
    target_idx = -1
    position_idx = -2
    label_idx = -3
    estimator.train()

    num_corrects = 0
    loss_sum = 0
    batch_count = 0
    sample_count = 0

    if TPU:
        trn_iterator = pl.ParallelLoader(trn_iterator,
                                         [device]).per_device_loader(device)
    tbar = tqdm(trn_iterator, ncols=80)
    for batch in tbar:
        inputs = {}
        for i, feat in enumerate(FEATURES):
            tens = torch.Tensor(batch[i]).to(device)
            if DTYPES[feat] is int:
                inputs[feat] = tens.long()
            elif DTYPES[feat] is float:
                inputs[feat] = tens.float()
            elif DTYPES[feat] is bool:
                inputs[feat] = tens.bool()
        inputs['targets'] = torch.Tensor(batch[target_idx]).to(
            device).long()
        inputs['positions'] = torch.Tensor(batch[position_idx]).to(
            device).long()
        labels = torch.Tensor(batch[label_idx]).to(
            device).float()

        optim.zero_grad()
        output = estimator(inputs=inputs)
        mask = inputs['targets'] != PAD_VALUE
        output = torch.masked_select(output, mask)
        labels = torch.masked_select(labels, mask)

        loss = criterion(output, labels)
        loss.backward()
        if TPU:
            xm.optimizer_step(optim)
        else:
            optim.step()

        loss_sum += loss.item()
        pred = (output >= 0.5).long()
        num_corrects += (pred == labels).sum().item()
        batch_count += 1
        sample_count += len(labels)

        tbar.set_description(
            'trn_loss - {:.4f}'.format(loss_sum / batch_count))

    acc = num_corrects / sample_count
    loss = loss_sum / batch_count

    return loss, acc


def val_epoch(estimator, val_iterator, criterion, device="cpu"):
    target_idx = -1
    position_idx = -2
    label_idx = -3
    estimator.eval()

    loss_sum = 0
    batch_count = 0
    num_corrects = 0
    sample_count = 0
    truth = np.empty(0)
    outs = np.empty(0)

    if TPU:
        val_iterator = pl.ParallelLoader(val_iterator,
                                         [device]).per_device_loader(device)
    tbar = tqdm(val_iterator, ncols=80)
    for batch in tbar:
        inputs = {}
        for i, feat in enumerate(FEATURES):
            tens = torch.Tensor(batch[i]).to(device)
            if DTYPES[feat] is int:
                inputs[feat] = tens.long()
            elif DTYPES[feat] is float:
                inputs[feat] = tens.float()
            elif DTYPES[feat] is bool:
                inputs[feat] = tens.bool()
        inputs['targets'] = torch.Tensor(batch[target_idx]).to(
            device).long()
        inputs['positions'] = torch.Tensor(batch[position_idx]).to(
            device).long()
        labels = torch.Tensor(batch[label_idx]).to(
            device).float()

        with torch.no_grad():
            output = estimator(inputs=inputs)  # [:, -1]
        mask = inputs['targets'] != PAD_VALUE
        output = torch.masked_select(output, mask)
        labels = torch.masked_select(labels, mask)

        loss = criterion(output, labels)
        loss_sum += loss.item()
        batch_count += 1

        pred = (output >= 0.5).long()
        num_corrects += (pred == labels).sum().item()
        sample_count += len(labels)
        truth = np.r_[truth, labels.view(-1).data.cpu().numpy()]
        outs = np.r_[outs, output.view(-1).data.cpu().numpy()]

        tbar.set_description(
            'val_loss - {:.4f}'.format(loss_sum / batch_count))

    acc = num_corrects / sample_count
    auc = roc_auc_score(truth, outs)
    loss = loss_sum / batch_count

    return loss, acc, auc


def train_transformer(estimator_type, estimator_params, epochs, batch_size,
                      device, early_stopping, eps, nworkers, cont, debug,
                      debug_size, data_size, valid_size):
    train = load_data('train_merged.parquet')
    print('Using Columns -', list(train.columns))
    print('total size:', train.shape)
    _, train = split_train_valid(train, debug_size if debug else data_size)
    print('data size:', train.shape,
          '- num users:', train['user_id'].nunique())
    train = preprocess(train)
    train = preprocess_timestamps(train)
    train, valid = split_train_valid(train, valid_size)
    print('train size:', train.shape,
          '- num users:', train['user_id'].nunique())
    print('valid size:', valid.shape,
          '- num users:', valid['user_id'].nunique())
    gc.collect()

    last_auc = 0
    if cont and not debug:
        estimator, optim, sched, last_auc = \
            load_model(MODEL, for_training=True, warmup_steps=WARMUP_STEPS)
        print('Previous Validation AUC:', last_auc)
    else:
        estimator = create_model(estimator_type, **estimator_params)
        optim = create_optimizer(estimator, lr=LEARNING_RATE)
        sched = create_scheduler(estimator, optim, warmup_steps=WARMUP_STEPS)

    trn_dataset = Riiid(dt=train, seq_len=SEQ_LEN, pad_value=PAD_VALUE)
    val_dataset = Riiid(dt=valid, seq_len=SEQ_LEN, pad_value=PAD_VALUE)

    if TPU:
        trn_sampler = torch.utils.data.distributed.DistributedSampler(
            trn_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=True)

        val_sampler = torch.utils.data.distributed.DistributedSampler(
            val_dataset,
            num_replicas=xm.xrt_world_size(),
            rank=xm.get_ordinal(),
            shuffle=False)

    trn_dataloader = torch.utils.data.DataLoader(
        dataset=trn_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=nworkers,
        sampler=trn_sampler if TPU else None,
        shuffle=False if TPU else True)
    val_dataloader = torch.utils.data.DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=nworkers,
        sampler=val_sampler if TPU else None,
        shuffle=False)

    criterion = nn.BCELoss().to(device)
    over_fit = 0
    seed_everything(RANDOM_SEED)
    for epoch in range(epochs):
        print(f'\nEpoch {epoch + 1} - Learning rate:',
              optim.param_groups[0]['lr'])
        trn_loss, trn_acc = train_epoch(estimator, trn_dataloader,
                                        optim, criterion, device)
        print("  Training - loss:{:.4f} - acc: {:.4f}" \
              .format(trn_loss, trn_acc))

        val_loss, val_acc, val_auc = val_epoch(estimator, val_dataloader,
                                               criterion, device)
        color = '\033[91m' if val_auc > last_auc and epoch != 0 else ''
        print(color + "  Validation - loss: {:.4f} - acc: {:.4f}, auc: {:.6f}" \
              .format(val_loss, val_acc, val_auc) + '\033[0m')
        if sched is not None:
            if isinstance(sched, torch.optim.lr_scheduler.ReduceLROnPlateau):
                sched.step(val_auc)
            else:
                sched.step()

        if val_auc > last_auc + eps:
            last_auc = val_auc
            over_fit = 0
            save_model(estimator, optim, sched, val_auc)
        else:
            over_fit += 1

        if over_fit >= early_stopping:
            print("early stop epoch ", epoch + 1)
            break

    return estimator


def tpu_map_fn(index, flags):
    train_transformer(flags)


# ========================== TEST =======================================


class RiiidTest(torch.utils.data.Dataset):
    def __init__(self, dt, queries, seq_len, pad_value=0, local=False):
        self.data = dt
        self.dcols = {col: i for i, col in enumerate(
            [KEY_FEATURE] + FEATURES + [TARGET, 'timestamp_raw', 'position'])}

        self.queries = queries.copy()
        self.groups = None
        if local:
            self.groups = queries.groupby(KEY_FEATURE) \
                .apply(lambda r: r.values).values.tolist()
        self.seq_len = seq_len
        self.pad_value = pad_value
        self.is_local = local

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        if self.is_local:
            random.shuffle(self.groups)
            query = self.groups[0][[0]]
            query_label = query[0, self.dcols[TARGET]]
            query = np.delete(query, self.dcols[TARGET], axis=1)
            self.groups[0] = self.groups[0][1:]
            if self.groups[0].shape[0] == 0:
                self.groups = self.groups[1:]
        else:
            query = self.queries[[idx]]
        uid = query[0, self.dcols[KEY_FEATURE]]

        if uid in self.data:
            inputs = self.data[uid][-self.seq_len + 1:]
            labels = inputs[:, self.dcols[TARGET]]
            last_timestamp = inputs[-1, self.dcols['timestamp_raw']]
            curr_timestamp = query[0, self.dcols['timestamp']]
            diff_timestamp = np.ceil((curr_timestamp - last_timestamp) / 60000)
            query[0, self.dcols['timestamp']] = diff_timestamp
            last_pos = inputs[-1, self.dcols['position']]
            query = np.c_[query, last_pos + 1]
            query = np.delete(query, self.dcols[KEY_FEATURE], axis=1)
            inputs = np.delete(inputs, [self.dcols[KEY_FEATURE],
                                        self.dcols[TARGET],
                                        self.dcols['timestamp_raw']], axis=1)
            inputs = np.r_[inputs, query]
        else:
            query[0, self.dcols['timestamp']] = DEFAULT_VALUES['timestamp']
            last_pos = 0
            query = np.c_[query, last_pos + 1]
            query = np.delete(query, self.dcols[KEY_FEATURE], axis=1)

            inputs = query
            labels = np.empty(0)

        targets = labels + 1
        inputs = pad_batch(inputs, self.seq_len, self.pad_value)
        targets = pad_batch(targets, self.seq_len, self.pad_value)

        if self.is_local:
            labels = np.r_[labels, query_label]
            labels = pad_batch(labels, self.seq_len,
                               self.pad_value)
            uids = np.full(inputs.shape[0], uid)
            inputs = np.c_[inputs, uids, labels]

        return np.c_[inputs[1:], targets[:-1]].astype(np.float32)


def collate_fn_test(batch):
    return np.array(batch).transpose(2, 0, 1)


def update_stats(prev_data, prev_batch):
    dcols = {col: i for i, col in
             enumerate([KEY_FEATURE] + FEATURES + [TARGET, 'timestamp_raw',
                                                   'position'])}

    def update_stat(trow):
        uid = trow[0]
        if uid in prev_data:
            last_timestamp = prev_data[uid][-1, dcols['timestamp_raw']]
            curr_timestamp = trow[dcols['timestamp']]
            diff_timestamp = np.ceil((curr_timestamp - last_timestamp) / 60000)
            trow[dcols['timestamp']] = diff_timestamp
            trow = np.r_[
                trow, curr_timestamp]  # Add timestamp_raw of the query.
            last_pos = prev_data[uid][-1, dcols['position']]
            trow = np.r_[trow, last_pos + 1]
            prev_data[uid] = np.r_[prev_data[uid][-SEQ_LEN + 2:], [trow]] \
                .astype(int)
        else:
            curr_timestamp = trow[dcols['timestamp']]
            trow[dcols['timestamp']] = DEFAULT_VALUES['timestamp']
            trow = np.r_[
                trow, curr_timestamp]  # Add timestamp_raw of the query.
            last_pos = 0
            trow = np.r_[trow, last_pos + 1]
            prev_data[uid] = np.array([trow])

    np.apply_along_axis(update_stat, arr=prev_batch, axis=1)


def predict_local(estimator, filename, prev_data, is_debug, batch_size):
    test_set = preprocess(load_data(filename))

    if is_debug:
        _, test_set = split_train_valid(test_set, 0.1)
    print('test shape:', test_set.shape)
    test_dataset = RiiidTest(prev_data, test_set, SEQ_LEN, local=True)
    test_dataloader = \
        torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn_test,
                                    num_workers=0)
    preds = predict_test(estimator,
                         test_dataloader,
                         local=True,
                         prev_data=prev_data,
                         device=DEVICE)
    return preds


def predict_test(estimator,
                 tst_iterator,
                 local=False,
                 prev_data=None,
                 device="cpu"):
    target_idx = -1
    position_idx = -2
    estimator.eval()

    truth = np.empty(0)
    outs = np.empty(0)
    if local:
        tst_iterator = tqdm(tst_iterator, ncols=80)
    for batch in tst_iterator:
        inputs = {}
        for i, feat in enumerate(FEATURES):
            if DTYPES[feat] is int:
                inputs[feat] = torch.Tensor(batch[i].astype(int)) \
                    .to(device).long()
            elif DTYPES[feat] is float:
                inputs[feat] = torch.Tensor(batch[i]).to(device).float()
            elif DTYPES[feat] is bool:
                inputs[feat] = torch.Tensor(batch[i].astype(bool)) \
                    .to(device).bool()
        inputs['targets'] = torch.Tensor(batch[target_idx].astype(np.int64)) \
            .to(device).long()
        inputs['positions'] = torch.Tensor(batch[position_idx]).to(
            device).long()

        with torch.no_grad():
            output = estimator(inputs=inputs)[:, -1]
        outs = np.r_[outs, output.data.cpu().numpy()]

        if local:
            prev_batch = batch[[-3] + list(range(len(FEATURES))) + [-2]]
            prev_batch = prev_batch[:, :, -1].T
            update_stats(prev_data, prev_batch)

            labels = batch[-2, :, -1]
            truth = np.r_[truth, labels]
            tst_iterator.set_description(
                'test_auc - {:.4f}'.format(roc_auc_score(truth, outs)))

    return outs


def predict_submission_group(estimator,
                             tst_batch,
                             prev_batch,
                             prev_data,
                             batch_size=128):
    all_cols = list(tst_batch.columns) + ADDED_FEATURES + [TARGET]
    all_cols = dict(zip(all_cols, range(len(all_cols))))
    used_cols = [all_cols[feat] for feat in [KEY_FEATURE] + FEATURES]
    tst_batch = tst_batch.values

    if (prev_batch is not None) & (psutil.virtual_memory().percent < 95):
        prev_batch = np.c_[prev_batch, eval(
            tst_batch[0, all_cols['prior_group_answers_correct']])]
        prev_batch = prev_batch[prev_batch[:, all_cols['content_type_id']] == 0
                                ][:, used_cols + [all_cols[TARGET]]]
        update_stats(prev_data, prev_batch)

    default_values = [DEFAULT_VALUES[feat] for feat in ADDED_FEATURES]
    question_feats = np.apply_along_axis(
        lambda rid: [QUESTIONS_DF[rid[0]][feat] for feat in ADDED_FEATURES]
        if rid[0] in QUESTIONS_DF else default_values,
        axis=1, arr=tst_batch[:, [all_cols['content_id']]])
    tst_batch = np.c_[tst_batch, question_feats]
    tst_batch = preprocess_test(tst_batch, all_cols)
    prev_batch = tst_batch.copy()

    qrows = tst_batch[:, all_cols['content_type_id']] == 0
    tst_batch = tst_batch[qrows]
    tst_dataset = RiiidTest(prev_data, tst_batch[:, used_cols], SEQ_LEN)
    tst_dataloader = torch.utils.data.DataLoader(dataset=tst_dataset,
                                                 batch_size=batch_size,
                                                 collate_fn=collate_fn_test,
                                                 num_workers=0)

    _preds = predict_test(estimator, tst_dataloader, device=DEVICE)
    tst_batch = np.c_[tst_batch, _preds]
    _predictions = pd.DataFrame(
        tst_batch[:, [all_cols[col] for col in SUBMISSION_COLUMNS]],
        columns=SUBMISSION_COLUMNS)

    return {'preds': _predictions,
            'prev_batch': prev_batch,
            'prev_data': prev_data}


def load_previous_data(pdata_path):
    #     key_feature_col = 0
    #     pdata = preprocess(load_data(pdata_path)).to_records(index=False)
    #     pdata = np.sort(pdata, order=[KEY_FEATURE, 'position'])
    #     print('here')
    #     pdata = np.split(pdata, np.unique(pdata[:, key_feature_col], return_index=True)[1][1:])
    #     pdata = {di[0, key_feature_col]: di[-SEQ_LEN + 1:].astype(int) for di in pdata}

    pdata = preprocess(load_data(pdata_path))
    gc.collect()
    print(f'getting {SEQ_LEN} long sequences...')
    pdata = pdata.groupby(KEY_FEATURE).tail(SEQ_LEN)
    pdata = preprocess_timestamps(pdata)
    pdata = pdata.groupby(KEY_FEATURE).apply(lambda g: g.values)
    gc.collect()
    return pdata

# %% code[]
"""# Train"""

MODEL = TransformerEncoderModel2  # @param ["TransformerModel", "TransformerModel2", "TransformerModel3", "TransformerEncoderModel", "TransformerEncoderModel2"] {type:"raw"}

MODEL_FILENAME = f'{MODEL.__name__}_best.pth'  # @param {type:"string"}


# @markdown # Model Settings

SEQ_LEN = 256  # @param {type:"integer"}

D_MODEL = 128  # @param {type:"integer"}

NHEAD = 8  # @param {type:"integer"}

N_ENC_LAYERS = 2  # @param {type:"integer"}

N_DEC_LAYERS = 2  # @param {type:"integer"}

DIM_FEEDFORWARD = 1028  # @param {type:"integer"}

DROPOUT = 0.1  # @param {type:"slider", min:0, max:1, step:0.05}

ACTIVATION = "gelu"  # @param ["relu", "gelu"]

LEARNING_RATE = 0.0005  # @param {type:"number"}


# @markdown # Training Settings

EPOCHS = 100  # @param {type:"integer"}

BATCH_SIZE = 64  # @param {type:"integer"}

EARLY_STOPPING = 15  # @param {type:"integer"}

NUM_WORKERS = 0  # @param {type:"integer"}

SCHED_TYPE = "plateau"  # @param ["plateau", "lambda", "NONE"]

WARMUP_STEPS = 20  # @param {type:"integer"}




PAD_VALUE = 0
START_TOKEN = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu") \
    if not TPU else xm.xla_device()


# %% code[]


RETRAIN = True  # @param {type:"boolean"}

CONTINUE = False  # @param {type:"boolean"}

DEBUG = False  # @param {type:"boolean"}

DATA_SIZE = 1  # @param {type:"slider", min:0.01, max:1, step:0.01}

DEBUG_SIZE = 0.1  # @param {type:"slider", min:0.01, max:1, step:0.01}

LOCAL_SAMPLE = False  # @param {type:"boolean"}


# %% code[]


if __name__ == '__main__':
    print('Using Device -', DEVICE)
    print('Using Model -', MODEL.__name__)
    model_params = {
        'seq_len': SEQ_LEN,
        'd_model': D_MODEL,
        'nhead': NHEAD,
        'num_encoder_layers': N_ENC_LAYERS,
        'num_decoder_layers': N_DEC_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'activation': ACTIVATION
    }

    train_params = {
        'estimator_type': MODEL,
        'estimator_params': model_params,
        'epochs': EPOCHS,
        'batch_size': BATCH_SIZE,
        'early_stopping': EARLY_STOPPING,
        'nworkers': NUM_WORKERS,
        'device': DEVICE,
        'eps': 1e-5,
        'cont': CONTINUE,
        'debug': DEBUG,
        'data_size': DATA_SIZE,
        'debug_size': DEBUG_SIZE,
        'valid_size': 0.1
    }

    if RETRAIN and not COMPILE:
        if TPU:
            xmp.spawn(tpu_map_fn, args=(train_params,), nprocs=8, start_method='fork')
        else:
            model = train_transformer(**train_params)

"""# Test"""

# ============================= TESTING ===================================
print('Testing...')

if LOCAL_SAMPLE and not COMPILE:
    print('predicting on local sample...')
    model, _, _, _ = load_model(MODEL)
    model.eval()

    predictions = predict_local(model,
                                'test_transformer.parquet',
                                pd.Series(),
                                is_debug=DEBUG,
                                batch_size=32 if DEBUG else 512)
else:
    print('loading previous data...')
    previous_data = load_previous_data('train_merged.parquet')
    print('loading question data...')
    QUESTIONS_DF = pd.read_parquet(PARQUETS_DIR + 'questions.parquet').to_dict(
        'index')
    model, _, _, _ = load_model(MODEL)
    model.eval()
    gc.collect()

    if not COMPILE:
        print('Submitting locally...')
        previous_batch = None
        tgts = []
        example_test = pd.read_csv(DATA_DIR + 'example_test.csv')
        submission = pd.DataFrame(columns=SUBMISSION_COLUMNS)
        for gnum in tqdm(example_test['group_num'].unique()):
            test_batch = example_test[
                example_test['group_num'] == gnum].copy()
            # test_batch['content_type_id'] = np.random.randint(0, 2, len(test_batch))
            # test_batch['user_id'] = 1931258865  # np.random.randint(0, previous_data.index.max() + 10000, len(test_batch))
            # test_batch['content_id'] = 10542131233  # np.random.randint(0, 20000, len(test_batch))
            preds, previous_batch, previous_data = predict_submission_group(
                model, test_batch,
                previous_batch,
                previous_data,
                batch_size=1024).values()
            tgts.extend(
                eval(test_batch['prior_group_answers_correct'].iloc[0]))
            submission = submission.append(preds)
        tgts.extend([-1] * len(test_batch))
        submission['target'] = tgts
        submission['pred'] = (submission[TARGET] >= 0.5).astype(np.int8)
        submission = submission.reset_index(drop=True)
        acc = submission[submission['target'] != -1]['target'] == \
              submission[submission['target'] != -1]['pred']
        acc = sum(acc) / len(acc)
        print(submission)
        print('Accuracy', acc)
    else:
        print('Submitting...')
        env = riiideducation.make_env()
        previous_batch = None
        for test_batch, _ in env.iter_test():
            preds, previous_batch, previous_data = \
                predict_submission_group(model,
                                         test_batch,
                                         previous_batch,
                                         previous_data,
                                         batch_size=1024).values()
            env.predict(preds)