
import torch
import numpy as np

import os
from datetime import datetime

import neptune
import torch
from torch import nn
from torch.nn.modules.loss import BCEWithLogitsLoss
from torch import optim
import optuna
import neptunecontrib.monitoring.optuna as opt_utils
import argparse

import utils


# torch.manual_seed(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAINING_BATCH_NUM = 100000
VAL_BATCH_NUM = 500
BATCH_SIZE = 20
VOCAB_SIZE = 2

HIDDEN_SIZE = 200

VAL_INTERVAL = 500
LOG_INVERVAL = 1


class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(RNN, self).__init__()
        self.act = torch.nn.Tanh()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.W_hh = nn.Parameter(utils.rand_ortho((self.hidden_dim, self.hidden_dim),
                np.sqrt(6. / (self.hidden_dim + self.hidden_dim))))
        self.W_xh = nn.Parameter(torch.empty(self.in_dim, self.hidden_dim))
        torch.nn.init.normal_(self.W_xh, 0, 0.1)
        self.W_hy = nn.Parameter(torch.empty(self.hidden_dim, self.out_dim))
        torch.nn.init.normal_(self.W_hy, 0, 0.1)
        self.b_h = nn.Parameter(torch.zeros([self.hidden_dim]))
        self.b_y = nn.Parameter(torch.zeros([self.out_dim]))

    def forward(self, x):
        seq_len, batch_size, _ = x.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        out_logits = []
        for step in range(seq_len):
            current_x = x[step, :, :]
            h = self.act(
                torch.mm(current_x, self.W_xh) + torch.mm(h, self.W_hh) + self.b_h)
            out_logits.append(torch.mm(h, self.W_hy) + self.b_y)
        return torch.stack(out_logits, dim=0).squeeze_()


def validate(model, val_list):
    criteron = torch.nn.BCEWithLogitsLoss()
    with torch.no_grad():
        total_loss = 0
        total_batch = 0
        total_correct = 0
        for i, (X, Y) in enumerate(val_list):
            logit_out = model(X)
            num_correct = utils.num_correct_samples(logit_out, Y)
            loss = criteron(torch.transpose(logit_out, 0, 1), torch.transpose(Y.float(), 0, 1))
            total_loss += loss
            total_batch += 1
            total_correct += num_correct
        avg_loss = total_loss/total_batch
        batch_size = val_list[0][0].size(1)
        num_batch = len(val_list)
        total_samples = batch_size * num_batch
        percent_correct = total_correct/total_samples
    return avg_loss, percent_correct


def run_training():
    parser = argparse.ArgumentParser(description="Training a simple RNN model")
    parser.add_argument('--seq_len', help='sequence length',
                        type=int, required=True)
    parser.add_argument('--lr', help='SGD learning rate',
                        type=float, required=True)
    parser.add_argument('--momentum', help='SGD momentum',
                        type=float, required=True)
    args = parser.parse_args()
    seq_len = args.seq_len
    lr = args.lr
    momentum = args.momentum

    dataloaders = utils.get_dataloaders(
        BATCH_SIZE,
        seq_len,
        train_batch_num=TRAINING_BATCH_NUM,
        val_batch_num=VAL_BATCH_NUM)
    train_loader = dataloaders['train']
    val_list = dataloaders['val']

    model = RNN(VOCAB_SIZE, HIDDEN_SIZE, 1).to(DEVICE)
    model.train()

    criterion = torch.nn.BCEWithLogitsLoss()
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr, momentum=momentum, nesterov=True)

    neptune.init(project_qualified_name="peterpdai/test")
    neptune.create_experiment(
        name=f'rnn_training_seq_len_{seq_len:02d}',
        upload_source_files=['*.py'],
        tags=["rnn", "training"])

    for i, (X, Y) in enumerate(train_loader):
        optimizer.zero_grad()
        logits = model(X)
        loss = criterion(torch.transpose(logits, 0, 1), torch.transpose(Y.float(), 0, 1))
        loss.backward()
        optimizer.step()

        if (i+1) % LOG_INVERVAL == 0:
            neptune.log_metric("training_loss", i, loss)

        if (i+1) % VAL_INTERVAL == 0:
            val_loss, percent_correct = validate(model, val_list)
            print(
                f"iteration {i+1}   train_loss {loss.item():.4f}    val_loss {val_loss:.4f}     \
                    percent_correct {percent_correct:.4f}")
            neptune.log_metric("validation_loss", i, val_loss)
            neptune.log_metric("validation_accuracy", i, percent_correct)


# def objective(trial):

#     # Hyperparameters to search
#     seq_len = trial.suggest_int("seq_len", SEQ_LEN, SEQ_LEN)
#     lr = trial.suggest_float("lr", 1e-4, 1, log=True)
#     momentum = trial.suggest_float("momentum", 0.1, 0.99)

#     # Prepare data
#     dataloaders = utils.get_dataloaders(
#         BATCH_SIZE,
#         seq_len,
#         train_batch_num=TRAINING_BATCH_NUM,
#         val_batch_num=VAL_BATCH_NUM)
#     train_loader = dataloaders['train']
#     val_list = dataloaders['val']

#     rnn = RNN(4, HIDDEN_SIZE, 1).to(DEVICE)

#     criterion = torch.nn.BCEWithLogitsLoss()
#     optimizer = torch.optim.SGD(rnn.parameters(), lr=lr, momentum=momentum, nesterov=True)

#     # Training loop
#     best_acc = 0

#     for i, (X, Y) in enumerate(train_loader):
#         rnn.train()
#         optimizer.zero_grad()
#         logits = rnn(X)
#         loss = criterion(torch.transpose(logits, 0, 1), torch.transpose(Y.float(), 0, 1))
#         loss.backward()
#         optimizer.step()

#         if (i+1) % VAL_INTERVAL == 0:
#             rnn.eval()
#             val_loss, percent_correct = validate(rnn, val_list)

#             if percent_correct > best_acc:
#                 best_acc = percent_correct

#             print(
#                 f"iteration {i}   train_loss {loss.item():.8f}    val_loss {val_loss:.8f}     percent_correct {percent_correct:.4f}")
#             trial.report(percent_correct, i)

#             if trial.should_prune():
#                 raise optuna.exceptions.TrialPruned()

#     return best_acc


# def run_hparameter_search():
#     parser = argparse.ArgumentParser(description="Hyperparameter search for a given sequence length")
#     parser.add_argument("seq_len", help="sequence length", type=int)
#     args = parser.parse_args()
#     SEQ_LEN = args.seq_len

#     proj = neptune.init(project_qualified_name="peterpdai/tarseq-hparams-search")
#     exp = proj.create_experiment(name=f'optuna-sweep_length_{SEQ_LEN:02d}',
#                                  upload_source_files=['tarseq_hsearch.py', 'utils.py'],
#                                  tags=["rnn", "hyperparameter-search"])

#     neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)

#     study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
#     study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

#     pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
#     complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

#     print("Study statistics: ")
#     print("  Number of finished trials: ", len(study.trials))
#     print("  Number of pruned trials: ", len(pruned_trials))
#     print("  Number of complete trials: ", len(complete_trials))

#     print("Best trial:")
#     trial = study.best_trial

#     print("  Value: ", trial.value)

#     print("  Params: ")
#     for key, value in trial.params.items():
#         print("    {}: {}".format(key, value))


if __name__ == "__main__":
    run_training()
