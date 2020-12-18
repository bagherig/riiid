#!/usr/bin/env python
# coding: utf-8

# %% [code]

import os
import gc
import math
import random
import warnings

import numpy as np
import pandas as pd
import psutil
import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from tqdm import tqdm

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

warnings.filterwarnings('ignore')
gc.enable()

# %% [code]

COMPILE = 0
VERSION = 'v1'

PARENT_DIR = os.path.dirname(__file__) + '/../'
DATA_DIR = PARENT_DIR + '/data/riiid-test-answer-prediction/'
PARQUETS_DIR = PARENT_DIR + 'data/parquets/'
MODELS_DIR = PARENT_DIR + 'models/'

OUT_DIR = PARENT_DIR + 'temp/'
# %% [code]

TARGET = 'answered_correctly'
KEY_FEATURE = 'user_id'
FEATURES = [
    'content_id',
    'prior_question_elapsed_time',
    'part',
]

DTYPES = {
    'content_id': int,
    'prior_question_elapsed_time': int,
    # 'prior_question_had_explanation': bool,
    'part': int,
}

ADDED_FEATURES = [
    'part'
]

SUBMISSION_COLUMNS = [
    'row_id',
    TARGET
]

# %% [code]
# ========================== Transformer Model ================================

def load_data(filename):
    return pd.read_parquet(PARQUETS_DIR + filename,
                           columns=[KEY_FEATURE] + FEATURES + [
                               TARGET])  # .iloc[-1_000_000:]


def split_train_valid(dt, val_fraction):
    val_size = 0
    trn_size = 0
    val_uids = []
    n_samples_per_user = dt.groupby(KEY_FEATURE)[
        TARGET].count().sort_values().reset_index().values.tolist()
    while n_samples_per_user:
        uid, nsamples = n_samples_per_user.pop()
        if trn_size * val_fraction > val_size:
            val_uids.append(uid)
            val_size += nsamples
        else:
            trn_size += nsamples

    val = dt[dt[KEY_FEATURE].isin(val_uids)]
    trn = dt.drop(val.index)
    return trn, val


def preprocess(dt):
    dt["prior_question_elapsed_time"] = dt["prior_question_elapsed_time"].mask(
        dt["prior_question_elapsed_time"] == 0).fillna(26000)
    dt["prior_question_elapsed_time"] = np.ceil(
        dt["prior_question_elapsed_time"] / 1000).astype(np.int16)
    dt["content_id"] += 1
    #     data["answered_correctly"] += 1
    return dt


def pad_batch(x, window_size, pad_value=0):
    shape = ((0, window_size - x.shape[0]),) + tuple(
        (0, 0) for i in range(len(x.shape) - 1))
    return np.pad(x, shape, constant_values=pad_value)


def rolling_window(a, w):
    s0, s1 = a.strides
    m, n = a.shape
    return np.lib.stride_tricks.as_strided(
        a,
        shape=(m - w + 1, w, n),
        strides=(s0, s0, s1))


def make_time_series(x, windows_size, pad_value=0):
    x = np.pad(x, [[0, windows_size - 1], [0, 0]], constant_values=pad_value)
    x = rolling_window(x, windows_size)
    return x


def create_scheduler(estimator, optim, warmup_steps=10, last_epoch=-1):
    # lr_lambda = lambda epoch: 1e-3
    lr_lambda = lambda epoch: (estimator.d_model ** (-0.5) *
                               min((epoch + 1) ** (-0.5),
                                   (epoch + 1) * warmup_steps ** (-1.5)))
    return torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda=lr_lambda,
                                             last_epoch=last_epoch)


def create_optimizer(estimator, lr):
    return torch.optim.Adam(estimator.parameters(), lr=lr)


def create_model(**params):
    return TransformerModel(**params)


def save_model(estimator, optim, sched):
    checkpoint = {'model_params': estimator.params,
                  'model_state_dict': estimator.state_dict(),
                  'optimizer_state_dict': optim.state_dict(),
                  'learning_rate': optim.param_groups[0]['lr']
                  }
    if sched is not None:
        checkpoint = {**checkpoint,
                      'sheduler_state_dict': sched.state_dict(),
                      'epoch': sched.last_epoch
                      }

    torch.save(checkpoint, OUT_DIR + MODEL_FILENAME)

def load_model(for_training=False, warmup_steps=10):
    if os.path.exists(OUT_DIR + MODEL_FILENAME):
        filepath = OUT_DIR + MODEL_FILENAME
    else:
        filepath = MODELS_DIR + f'transformer/{VERSION}/{MODEL_FILENAME}'

    print(f'Loading model from {filepath}')
    checkpoint = torch.load(filepath, map_location=DEVICE)

    estimator = create_model(**checkpoint['model_params'])
    estimator.load_state_dict(checkpoint['model_state_dict'])
    estimator.to(DEVICE)

    optim = None
    sched = None
    if for_training:
        optim = create_optimizer(estimator, lr=checkpoint['learning_rate'])
        optim.load_state_dict(checkpoint['optimizer_state_dict'])

        if 'sheduler_state_dict' in checkpoint:
            sched = create_scheduler(estimator, optim, warmup_steps,
                                     last_epoch=checkpoint['epoch'])
            sched.load_state_dict(checkpoint['sheduler_state_dict'])

    return estimator, optim, sched


# %% [code]


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (
                -math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class TransformerModel(nn.TransformerEncoder):
    def __init__(self, **params):
        print(params)
        self.params = params
        self.d_model = params['d_model']
        self.seq_len = params.pop('seq_len', 96)
        self.n_layers = params.pop('n_layers', 4)
        self.dropout_rate = params['dropout']
        encoder_layer = nn.TransformerEncoderLayer(**params)
        encoder_norm = nn.LayerNorm(self.d_model)
        super(TransformerModel, self).__init__(encoder_layer=encoder_layer,
                                               num_layers=self.n_layers,
                                               norm=encoder_norm)
        self.max_exercise = 13523
        self.max_part = 7
        self.max_time = 300
        self.max_target = 2
        self.start_token = START_TOKEN

        # self.pos_embedding = nn.Embedding(self.d_model, self.d_model) # positional embeddings
        self.pos_embedding = PositionalEncoding(self.d_model,
                                                self.dropout_rate,
                                                self.seq_len)
        self.exercise_embeddings = \
            nn.Embedding(num_embeddings=self.max_exercise + 1,
                         embedding_dim=self.d_model)  # exercise_id
        self.part_embeddings = \
            nn.Embedding(num_embeddings=self.max_part + 1,
                         embedding_dim=self.d_model)
        self.elapsed_time_embeddings = (# nn.Linear(self.seq_len, self.d_model))
            nn.Embedding(num_embeddings=self.max_time + 1,
                         embedding_dim=self.d_model))
        self.target_embeddings = \
            nn.Embedding(num_embeddings=self.max_target + 2,
                         embedding_dim=self.d_model)
        self.decoder = nn.Linear(self.d_model, 1)
        self.linear = nn.Linear(self.seq_len, 1)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(self.dropout_rate)
        self.norm1 = nn.LayerNorm(self.d_model)

        self.device = DEVICE
        self.future_mask = self.generate_square_subsequent_mask(
            self.seq_len).to(self.device)
        self.init_weights()

    def generate_square_subsequent_mask(self, sz):
        mask = torch.triu(torch.ones(sz, sz), 1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask

    @staticmethod
    def make_pad_mask(inp):
        return inp == PAD_VALUE

    def set_start_token(self, inp):
        inp[:, 0] = self.start_token

    def init_weights(self):
        initrange = 0.1
        # init embeddings
        self.exercise_embeddings.weight.data.uniform_(-initrange, initrange)
        self.part_embeddings.weight.data.uniform_(-initrange, initrange)
        self.elapsed_time_embeddings.weight.data.uniform_(-initrange,
                                                          initrange)
        self.target_embeddings.weight.data.uniform_(-initrange, initrange)
        self.linear.bias.data.zero_()
        self.linear.weight.data.uniform_(-initrange, initrange)

    def forward(self, inputs: np.ndarray):
        """
        S is the sequence length, N the batch size and E the Embedding Dimension (number of features).
        src: (S, N, E)
        src_mask: (S, S)
        src_key_padding_mask: (N, S)
        padding mask is (N, S) with boolean True/False.
        SRC_MASK is (S, S) with float(’-inf’) and float(0.0).
        """
        content_ids, elapsed_times, parts, targets = \
            inputs['content_id'], inputs['prior_question_elapsed_time'], \
            inputs['part'], inputs['targets']
        self.set_start_token(targets)
        pad_mask = self.make_pad_mask(targets)

        content_ids[content_ids > self.max_exercise] = 0
        elapsed_times[elapsed_times > self.max_time] = self.max_time
        parts[parts > self.max_part] = 0

        # print(self.exercise_embeddings(content_ids).size())
        # print(self.elapsed_time_embeddings(elapsed_times).size())

        embedded_inp = (self.exercise_embeddings(content_ids)
                        + self.elapsed_time_embeddings(elapsed_times)
                        + self.part_embeddings(parts)
                        + self.target_embeddings(targets)
                        ) * np.sqrt(self.d_model)  # (N, S, E)
        embedded_inp = self.pos_embedding(embedded_inp.transpose(0, 1))
        embedded_inp = self.norm1(embedded_inp)

        output = super(TransformerModel, self).forward(
            src=embedded_inp,
            # mask=self.future_mask,
            src_key_padding_mask=pad_mask)
        output = self.norm1(output)
        output = self.dropout(output)
        output = self.decoder(output).squeeze(-1).transpose(1, 0)
        output = self.dropout(output)
        output = self.linear(output).squeeze(-1)
        output = self.activation(output)

        return output


# %% [code]


class Riiid(torch.utils.data.Dataset):
    def __init__(self, groups, seq_len, pad_value=0,
                 max_samples_per_user=None, for_training=True):
        self.groups = groups
        self.sample_cap = (100, 500)
        counts = groups.count()[TARGET]
        counts[counts < self.sample_cap[0]] = self.sample_cap[0]
        counts[counts > self.sample_cap[1]] = self.sample_cap[1]
        self.probs = counts / sum(counts)
        self.uids = list(groups.groups)
        self.seq_len = seq_len
        self.pad_value = pad_value
        self.max_samples_per_user = max_samples_per_user
        self.for_training = for_training

    def __len__(self):
        return len(self.uids)

    def __getitem__(self, idx):
        if self.for_training:
            uid = np.random.choice(self.uids, size=1, p=self.probs)[0]
        else:
            uid = self.uids[idx]
        g = self.groups.get_group(uid).copy()
        g['targets'] = (g[TARGET] + 1).shift(fill_value=START_TOKEN)
        rolling_data = make_time_series(g.values, self.seq_len,
                                        pad_value=PAD_VALUE)

        n_sequences = len(rolling_data)
        if self.max_samples_per_user is not None:
            if isinstance(self.max_samples_per_user, int):
                nsamples = min(self.max_samples_per_user, n_sequences)
            else:
                assert (0 < self.max_samples_per_user <= 1)
                nsamples = int(min(n_sequences, self.sample_cap[1]) *
                               self.max_samples_per_user)
            idx = np.random.choice(np.arange(n_sequences),
                                   nsamples, replace=False)
            rolling_data = rolling_data[idx]

        return rolling_data


def collate_fn(batch):
    return np.concatenate(batch).transpose(2, 0, 1)


# %% [code]


def train_epoch(estimator, train_iterator, optim, sched, criterion,
                device="cpu",
                batch_limit=128):
    estimator.train()

    tbar = tqdm(train_iterator, ncols=100)
    num_corrects = 0
    loss_sum = 0
    batch_count = 0
    sample_count = 0

    for batch in tbar:
        inputs = {}
        for i, feat in enumerate(FEATURES):
            if DTYPES[feat] is int:
                inputs[feat] = torch.Tensor(batch[i]).to(device).long()
            elif DTYPES[feat] is float:
                inputs[feat] = torch.Tensor(batch[i]).to(device).float()
            elif DTYPES[feat] is bool:
                inputs[feat] = torch.Tensor(batch[i]).to(device).bool()
        inputs['targets'] = torch.Tensor(batch[-1]).to(device).long()

        labels_all = torch.Tensor(batch[-2]).to(device).long()

        n_samples = len(labels_all)
        n_batches = int(np.ceil(n_samples / batch_limit))
        for nbatch in range(n_batches):
            optim.zero_grad()

            start_idx = nbatch * batch_limit
            end_idx = (nbatch + 1) * batch_limit
            targets = inputs['targets'][start_idx: end_idx].data.cpu().numpy()
            pred_col_idx = (targets != PAD_VALUE).cumsum(1).argmax(1)
            assert(PAD_VALUE not in targets[np.arange(targets.shape[0]),
                                            pred_col_idx])

            output = estimator(inputs={name: feat[start_idx: end_idx]
                                       for name, feat in inputs.items()})
            # print(output.shape, output)
            labels = labels_all[start_idx: end_idx].float()
            # output = output[np.arange(targets.shape[0]), pred_col_idx]
            labels = labels[np.arange(targets.shape[0]), pred_col_idx]
            # print('\n', labels.shape, labels)
            # print(output.shape, output)
            loss = criterion(output, labels)
            loss.backward()
            optim.step()
            if sched is not None:
                sched.step()

            loss_sum += loss.item()
            # pred = (torch.sigmoid(output) >= 0.5).long()
            pred = (output >= 0.5).long()
            num_corrects += (pred == labels).sum().item()
            batch_count += 1
            sample_count += len(labels)

            tbar.set_description(
                f'{nbatch + 1}/{n_batches} | ' + 'trn_loss - {:.4f}'.format(
                    loss_sum / batch_count))

    acc = num_corrects / sample_count
    loss = loss_sum / batch_count

    return loss, acc


# %% [code]


def val_epoch(estimator, val_iterator, criterion, device="cpu",
              batch_limit=128):
    estimator.eval()

    loss_sum = 0
    batch_count = 0
    num_corrects = 0
    sample_count = 0
    truth = np.empty(0)
    outs = np.empty(0)

    tbar = tqdm(val_iterator, ncols=100)
    for n_iter, batch in enumerate(tbar):
        inputs = {}
        for i, feat in enumerate(FEATURES):
            if DTYPES[feat] is int:
                inputs[feat] = torch.Tensor(batch[i]).to(device).long()
            elif DTYPES[feat] is float:
                inputs[feat] = torch.Tensor(batch[i]).to(device).float()
            elif DTYPES[feat] is bool:
                inputs[feat] = torch.Tensor(batch[i]).to(device).bool()
        inputs['targets'] = torch.Tensor(batch[-1]).to(device).long()
        labels_all = torch.Tensor(batch[-2]).to(device).long()

        n_samples = len(labels_all)
        n_batches = int(np.ceil(n_samples / batch_limit))
        for nbatch in range(n_batches):
            start_idx = nbatch * batch_limit
            end_idx = (nbatch + 1) * batch_limit
            targets = inputs['targets'][start_idx: end_idx].data.cpu().numpy()
            pred_col_idx = (targets != PAD_VALUE).cumsum(1).argmax(1)
            assert(PAD_VALUE not in targets[np.arange(targets.shape[0]),
                                            pred_col_idx])

            with torch.no_grad():
                output = estimator(inputs={name: feat[start_idx: end_idx]
                                           for name, feat in inputs.items()})
            labels = labels_all[start_idx: end_idx].float()
            # output = output[np.arange(targets.shape[0]), pred_col_idx]
            labels = labels[np.arange(targets.shape[0]), pred_col_idx]

            loss = criterion(output, labels)
            loss_sum += loss.item()
            batch_count += 1

            # pred = (torch.sigmoid(output) >= 0.5).long()
            pred = (output >= 0.5).long()
            num_corrects += (pred == labels).sum().item()
            sample_count += len(labels)
            truth = np.r_[truth, labels.view(-1).data.cpu().numpy()]
            outs = np.r_[outs, output.view(-1).data.cpu().numpy()]

            tbar.set_description(
                f'{nbatch + 1}/{n_batches} | ' + 'val_loss - {:.4f}'.format(
                    loss_sum / batch_count))

    acc = num_corrects / sample_count
    auc = roc_auc_score(truth, outs)
    loss = loss_sum / batch_count

    return loss, acc, auc


# %% [code]


def train_transformer(estimator, optim, sched, train, valid, epochs=10,
                      n_user_batches=32, batch_limit=128,
                      max_samples_per_user=100, device="cpu", early_stopping=2,
                      eps=1e-4, nworkers=4):
    trn_dataset = Riiid(groups=train.groupby('user_id')[FEATURES + [TARGET]],
                        seq_len=SEQ_LEN,
                        pad_value=PAD_VALUE,
                        max_samples_per_user=max_samples_per_user,
                        for_training=False)
    trn_dataloader = torch.utils.data.DataLoader(dataset=trn_dataset,
                                                 batch_size=n_user_batches,
                                                 collate_fn=collate_fn,
                                                 num_workers=nworkers)

    val_dataset = Riiid(groups=valid.groupby('user_id')[FEATURES + [TARGET]],
                        seq_len=SEQ_LEN, pad_value=PAD_VALUE,
                        max_samples_per_user=None,
                        for_training=False)
    val_dataloader = torch.utils.data.DataLoader(dataset=val_dataset,
                                                 batch_size=n_user_batches,
                                                 collate_fn=collate_fn,
                                                 num_workers=nworkers)

    criterion = nn.BCELoss()
    criterion.to(device)
    estimator.to(device)

    over_fit = 0
    last_auc = 0
    for epoch in range(epochs):
        trn_loss, trn_acc = train_epoch(estimator, trn_dataloader, optim,
                                        sched, criterion, device, batch_limit)
        print("Training epoch {} - loss:{:.4f} - acc: {:.4f}".format(epoch + 1,
                                                                     trn_loss,
                                                                     trn_acc))

        val_loss, val_acc, val_auc = val_epoch(estimator, val_dataloader,
                                               criterion, device, batch_limit)
        print(
            "Validation epoch {} - loss: {:.4f} - acc: {:.4f}, auc: {:.6f}"\
                .format(epoch + 1, val_loss, val_acc, val_auc))

        if val_auc > last_auc + eps:
            last_auc = val_auc
            over_fit = 0
            save_model(estimator, optimizer, sched)
        else:
            over_fit += 1

        if over_fit >= early_stopping:
            print("early stop epoch ", epoch + 1)
            break

    return estimator


# %% [code]
# ========================== TEST =======================================


class RiiidTest(torch.utils.data.Dataset):
    def __init__(self, dt, queries, seq_len, pad_value=0, local=False):
        self.data = dt
        self.dcols = {col: i for i, col in
                      enumerate([KEY_FEATURE] + FEATURES + [TARGET])}

        self.queries = queries
        self.groups = None
        if local:
            self.groups = queries.groupby(KEY_FEATURE)\
                .apply(lambda r: r.values).values.tolist()
        self.seq_len = seq_len
        self.pad_value = pad_value
        self.is_local = local

    def __len__(self):
        return len(self.queries)

    def __getitem__(self, idx):
        if self.is_local:
            random.shuffle(self.groups)
            query = self.groups[0][0]
            self.groups[0] = self.groups[0][1:]
            if self.groups[0].shape == 0:
                self.groups = self.groups[1:]
        else:
            query = self.queries[[idx]]
        uid = query[0, self.dcols[KEY_FEATURE]]
        query = np.delete(query, self.dcols[KEY_FEATURE], axis=1)

        if uid in self.data.index:
            encoder_data = self.data[uid]
            labels = encoder_data[:, self.dcols[TARGET]]
            inputs = np.delete(encoder_data, [self.dcols[KEY_FEATURE],
                                              self.dcols[TARGET]],
                               axis=1)
            inputs = np.r_[inputs, query]
        else:
            inputs = query
            labels = np.empty(0)

        inputs = pad_batch(inputs, self.seq_len, self.pad_value)
        targets = np.r_[START_TOKEN, labels + 1]
        targets = pad_batch(targets, self.seq_len, self.pad_value)

        return np.c_[inputs, targets]


def collate_fn_test(batch):
    return np.array(batch).transpose(2, 0, 1)


# %% [code]


def update_stats(prev_data, prev_batch):
    def update_stat(trow):
        uid = trow[0]
        if uid in prev_data.index:
            prev_data[uid] = np.r_[prev_data[uid][-SEQ_LEN + 2:], [trow]]\
                .astype(np.float32)
        else:
            prev_data[uid] = np.array([trow])

    np.apply_along_axis(update_stat, arr=prev_batch, axis=1)


# %% [code]


def predict_local(filepath, prev_data, is_debug, batch_size):
    test_set = pd.read_parquet(PARQUETS_DIR + filepath,)
                               # columns=[KEY_FEATURE] + FEATURES + [TARGET])
    if is_debug:
        test_set = test_set.iloc[-50_000:]
    test_dataset = RiiidTest(prev_data, test_set,
                             SEQ_LEN, local=True)
    test_dataloader = \
        torch.utils.data.DataLoader(dataset=test_dataset,
                                    batch_size=batch_size,
                                    collate_fn=collate_fn_test,
                                    num_workers=0)
    preds = predict_test(model,
                         test_dataloader,
                         local=True,
                         prev_data=prev_data,
                         device=DEVICE)
    print('Test AUC:', roc_auc_score(test_set[TARGET], preds))
    return preds


# %% [code]


def predict_test(estimator,
                 tst_iterator,
                 local=False,
                 prev_data=None,
                 device="cpu"):
    estimator.eval()

    truth = np.empty(0)
    outs = np.empty(0)
    if local:
        tst_iterator = tqdm(tst_iterator)
    for batch in tst_iterator:
        inputs = {}
        for i, feat in enumerate(FEATURES):
            if DTYPES[feat] is int:
                inputs[feat] = torch.Tensor(batch[i].astype(int))\
                    .to(device).long()
            elif DTYPES[feat] is float:
                inputs[feat] = torch.Tensor(batch[i]).to(device).float()
            elif DTYPES[feat] is bool:
                inputs[feat] = torch.Tensor(batch[i].astype(bool)).\
                    to(device).bool()

        inputs['targets'] = torch.Tensor(batch[-1].astype(np.int64)).to(device).long()

        with torch.no_grad():
            output = estimator(inputs=inputs)
        # output = torch.sigmoid(output)

        targets = inputs['targets'].data.cpu().numpy()
        pred_col_idx = (targets != PAD_VALUE).cumsum(1).argmax(1)
        assert (PAD_VALUE not in targets[np.arange(targets.shape[0]),
                                         pred_col_idx])

        # output = output[np.arange(inputs['targets'].shape[0]), pred_col_idx]
        outs = np.r_[outs, output.data.cpu().numpy()]

        if local:
            update_stats(prev_data, batch)
            labels = batch[-2]
            labels = labels[np.arange(inputs['targets'].shape[0]),
                            pred_col_idx]
            truth = np.r_[truth, labels]
            tst_iterator.set_description(
                'test_auc - {:.4f}'.format(roc_auc_score(truth, outs)))

    return outs


# %% [code]


def predict_group(estimator, tst_batch, prev_batch, prev_data, batch_size=128):
    all_cols = list(tst_batch.columns) + ADDED_FEATURES + [TARGET]
    all_cols = dict(zip(all_cols, range(len(all_cols))))
    used_cols = [all_cols[feat] for feat in [KEY_FEATURE] + FEATURES]

    tst_batch = preprocess(tst_batch).values
    if (prev_batch is not None) & (psutil.virtual_memory().percent < 90):
        # print(psutil.virtual_memory().percent)
        prev_batch = np.c_[prev_batch, eval(
            tst_batch[0, all_cols['prior_group_answers_correct']])]
        prev_batch = prev_batch[prev_batch[:, all_cols['content_type_id']] == 0
                                ][:, used_cols + [all_cols[TARGET]]]
        update_stats(prev_data, prev_batch)

    parts = np.apply_along_axis(
        lambda rid: TAGS_DF[rid[0]]['part'] if rid[0] in TAGS_DF else 0,
        axis=1, arr=tst_batch[:, [all_cols['content_id']]])
    tst_batch = np.c_[tst_batch, parts]
    prev_batch = tst_batch.copy()

    qrows = tst_batch[:, all_cols['content_type_id']] == 0
    tst_batch = tst_batch[qrows]
    tst_dataset = RiiidTest(prev_data, tst_batch[:, used_cols], SEQ_LEN)
    tst_dataloader = torch.utils.data.DataLoader(dataset=tst_dataset,
                                                 batch_size=batch_size,
                                                 collate_fn=collate_fn_test,
                                                 num_workers=0)

    _preds = predict_test(estimator, tst_dataloader, DEVICE)
    tst_batch = np.c_[tst_batch, _preds]
    _predictions = pd.DataFrame(
        tst_batch[:, [all_cols[col] for col in SUBMISSION_COLUMNS]],
        columns=SUBMISSION_COLUMNS)

    return {'preds': _predictions,
            'prev_batch': prev_batch,
            'prev_data': prev_data}


# %% [code]
# =============================================================================

MODEL_FILENAME = 'transformer_best2.pth' #@param {type:"string"}
SEQ_LEN = 96 #@param {type:"integer"}
D_MODEL = 512 #@param {type:"integer"}
NHEAD = 8 #@param {type:"integer"}
N_LAYERS = 4 #@param {type:"integer"}
DIM_FEEDFORWARD = 512 #@param {type:"integer"}
DROPOUT = 0.2 #@param {type:"slider", min:0, max:1, step:0.05}
ACTIVATION = "relu" #@param ["relu", "tanh", "sigmoid"]
LEARNING_RATE = 1e-3 #@param {type:"slider", min:1e-5, max:1, step:1e-5}
EPOCHS = 20 #@param {type:"integer"}
N_USER_BATCHES = 128 #@param {type:"integer"}
BATCH_LIMIT = 128 #@param {type:"integer"}
MAX_SAMPLES_PER_USER =  1#@param {type:"integer"}
EARLY_STOPPING = 10 #@param {type:"integer"}
WARMUP_STEPS = 50 #@param {type:"integer"}

PAD_VALUE = 0
START_TOKEN = 3
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# %% [code]
if __name__ == '__main__':
    print('Using Device -', DEVICE)
    retrain_transformer = 0 #@param {type:"boolean"}
    cont = 0 #@param {type:"boolean"}
    debug = 1 #@param {type:"boolean"}
    data_size = 20_000_000 #@param {type:"slider", min:1000000, max:100000000, step:1000000}
    debug_size = 5_000_000 #@param {type:"slider", min:100000, max:100000000, step:100000}

    model_params = {
        'seq_len': SEQ_LEN,
        'd_model': D_MODEL,
        'nhead': NHEAD,
        'n_layers': N_LAYERS,
        'dim_feedforward': DIM_FEEDFORWARD,
        'dropout': DROPOUT,
        'activation': ACTIVATION
    }

    if retrain_transformer and not COMPILE:
        data_path = 'train_transformer.parquet'
        data = preprocess(load_data(data_path))[-data_size:]
        if debug:
            data = data.iloc[-debug_size:]
        df_train, df_valid = split_train_valid(data, 0.05)
        print('train size:', df_train.shape[0], '- num users:',
              df_train['user_id'].nunique())
        print('valid size:', df_valid.shape[0], '- num users:',
              df_valid['user_id'].nunique())
        del data
        gc.collect()

        if cont:
            model, optimizer, scheduler = load_model(for_training=True,
                                                     warmup_steps=WARMUP_STEPS)
        else:
            model = create_model(**model_params)
            optimizer = create_optimizer(model, lr=LEARNING_RATE)
            scheduler = create_scheduler(model, optimizer,
                                         warmup_steps=WARMUP_STEPS)

        model = train_transformer(model, optimizer, scheduler,
                                  df_train, df_valid,
                                  epochs=EPOCHS,
                                  n_user_batches=N_USER_BATCHES,
                                  batch_limit=BATCH_LIMIT,
                                  max_samples_per_user=MAX_SAMPLES_PER_USER,
                                  early_stopping=EARLY_STOPPING,
                                  eps=1e-4,
                                  nworkers=0,
                                  device=DEVICE)

    # ============================= TESTING ===================================
    local_sample = True #@param {type:"boolean"}

    TAGS_DF = pd.read_parquet(PARQUETS_DIR + 'tags.parquet')
    # Add 1 to content ids to match embeddings.
    TAGS_DF.index = TAGS_DF.index + 1
    TAGS_DF = TAGS_DF.to_dict('index')
    pdata_path = 'train_merged.parquet' if COMPILE else \
                 'train_transformer.parquet'
    previous_data = preprocess(load_data(pdata_path)).groupby(
        KEY_FEATURE).apply(lambda g: g.tail(SEQ_LEN - 1).values)

# In[]
    model, _, _ = load_model()
    model.eval()

    if local_sample and not COMPILE:
        print('predicting on local sample...')
        predictions = predict_local('test_transformer.parquet',
                                    previous_data,
                                    is_debug=debug,
                                    batch_size=32)
    elif not COMPILE:
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
            preds, previous_batch, previous_data = predict_group(
                model, test_batch,
                previous_batch,
                previous_data,
                batch_size=1024).values()
            tgts.extend(eval(test_batch['prior_group_answers_correct'].iloc[0]))
            submission = submission.append(preds)
        tgts.extend([-1] * len(test_batch))
        submission['target'] = tgts
        submission['pred'] = (submission[TARGET] >= 0.5).astype(np.int8)
        submission = submission.reset_index(drop=True)
        print(submission)
    else:
        print('Submitting...')
        env = riiideducation.make_env()
        previous_batch = None
        for test_batch, _ in env.iter_test():
            preds, previous_batch, previous_data = \
                predict_group(model,
                              test_batch,
                              previous_batch,
                              previous_data,
                              batch_size=1024).values()
            env.predict(preds)