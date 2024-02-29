
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
TRAINING_BATCH_NUM = 10000
VAL_BATCH_NUM = 500
BATCH_SIZE = 20
VOCAB_SIZE = 4
COPY_NUM = 2

VAL_INTERVAL = 500
HIDDEN_SIZE = 150
alpha_i = 1
alpha_f = 0.00001
alpha_y = 0.0001
alpha_g = 0.01
f_update_prob = 0.05


def get_dataloaders(
        batch_size, seq_len, train_batch_num=100000, val_batch_num=500):
    # return a dict

    def val_data_list():
        return list(
            utils.batch_generator(
                seq_len, batch_size, 4, DEVICE, val_batch_num))
        # return [utils.batch_generator(seq_len, batch_size, 4, device, val_batch_num) for i in range(val_batch_num)]

    return {
        'train': utils.batch_generator(seq_len, batch_size, 4, DEVICE, train_batch_num),
        'val': val_data_list(),
    }

# def test_get_dataloader():
#     dataloaders = get_dataloaders(batch_size=20, seq_len=20, train_batch_num=1000,
#                                   val_batch_num=500)
#     train_loader = dataloaders['train']
#     val_loader = dataloaders['val']
#     train_batch_num = 0
#     val_batch_num = 0

#     train_X, train_Y = next(train_loader)

#     assert(train_X.shape == torch.Size([20, 20, 4]))
#     assert(train_X.dtype == torch.float)

#     assert(train_Y.shape == torch.Size([20, 20]))
#     assert(train_Y.dtype == torch.long)

#     val_X, val_Y = val_loader[0]

#     assert(val_X.shape == torch.Size([20, 20, 4]))
#     assert(val_X.dtype == torch.float)

#     assert(val_Y.shape == torch.Size([20, 20]))
#     assert(val_Y.dtype == torch.long)

#     for _ in train_loader:
#         train_batch_num += 1
#     for _ in val_loader:
#         val_batch_num += 1

#     assert (train_batch_num == (1000-1))
#     assert (val_batch_num == (500)) # list not generator


class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):

        super(RNN, self).__init__()

        # Set hyperparameters
        self.act = torch.nn.Tanh()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Manual RNN
        self.W_xh = nn.Parameter(
            utils.rand_ortho(
                (self.in_dim, self.hidden_dim),
                np.sqrt(6. / (self.in_dim + self.hidden_dim))))
        self.W_hh = nn.Parameter(
            utils.rand_ortho(
                (self.hidden_dim, self.hidden_dim),
                np.sqrt(6. / (self.hidden_dim + self.hidden_dim))))
        self.W_hy = nn.Parameter(
            utils.rand_ortho(
                (self.hidden_dim, self.out_dim),
                np.sqrt(6. / (self.hidden_dim + self.out_dim))))
        self.b_h = nn.Parameter(torch.zeros([self.hidden_dim]))
        self.b_y = nn.Parameter(torch.zeros([self.out_dim]))

    def forward(self, x):

        seq_len, batch_size, _ = x.shape

        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        out_logits = []
        for step in range(seq_len):
            current_x = x[step, :, :]  # [batch_size, vocab_size]
            h = self.act(
                torch.mm(current_x, self.W_xh) + torch.mm(h, self.W_hh) + self.b_h)
            out_logits.append(torch.mm(h, self.W_hy) + self.b_y)

        return torch.stack(out_logits, dim=0).squeeze_()


class TarSeq(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(TarSeq, self).__init__()

        # Set hyperparameters
        self.act = torch.nn.Tanh()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Forward parameters
        self.W_xh = nn.Parameter(
            utils.rand_ortho(
                (self.in_dim, self.hidden_dim),
                np.sqrt(6. / (self.in_dim + self.hidden_dim))))
        self.W_hh = nn.Parameter(
            utils.rand_ortho(
                (self.hidden_dim, self.hidden_dim),
                np.sqrt(6. / (self.hidden_dim + self.hidden_dim))))
        self.W_hy = nn.Parameter(
            utils.rand_ortho(
                (self.hidden_dim, self.out_dim),
                np.sqrt(6. / (self.hidden_dim + self.out_dim))))
        self.b_h = nn.Parameter(torch.zeros([self.hidden_dim]))
        self.b_y = nn.Parameter(torch.zeros([self.out_dim]))

        # Backward parameters

        self.V_hh = nn.Parameter(
            utils.rand_ortho(
                (self.hidden_dim, self.hidden_dim),
                np.sqrt(6. / (self.hidden_dim + self.hidden_dim))))
        self.c_h = nn.Parameter(torch.zeros([self.hidden_dim]))

    def forward(self, x):

        seq_len, batch_size, _ = x.shape

        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        out_logits = []
        for step in range(seq_len):
            current_x = x[step, :, :]  # [batch_size, vocab_size]
            h = self.act(
                torch.mm(current_x, self.W_xh) + torch.mm(h, self.W_hh) + self.b_h)
            out_logits.append(torch.mm(h, self.W_hy) + self.b_y)

        return torch.stack(out_logits, dim=0).squeeze_()


# def test_RNN():
#     in_dim = 4
#     hidden_dim = 100
#     out_dim = 1  # binary classification, 0 or 1
#     model = RNN(in_dim, hidden_dim, out_dim).to(torch.device)

#     X, Y = next(utils.batch_generator(seq_len=10, batch_size=3,
#                                       vocab_size=4, device=DEVICE, max_num_batch=100))

#     assert (X.shape == torch.Size([10, 3, 4]))
#     assert (X.dtype == torch.float32)
#     assert (Y.shape == torch.Size([10, 3]))
#     assert (Y.dtype == torch.int64)

#     logits = model(X)
#     assert (logits.shape == torch.Size([10, 3]))
#     assert (logits.dtype == torch.float32)

#     h_i = torch.zeros([3, 100], device=DEVICE)

#     for i in range(10):
#         h_i = torch.tanh(X[i, :, :] @ model.W_xh + h_i @ model.W_hh + model.b_h)
#         y_i = h_i @ model.W_hy + model.b_y
#         assert (torch.allclose(y_i.squeeze_(), logits[i, :]))

def get_criterion():
    return torch.nn.BCEWithLogitsLoss()


def get_optimizer(model_parameters):
    return torch.optim.SGD(
        model_parameters, lr=0.1, momentum=0.9, nesterov=True)


def num_correct_samples(logits, Y):
    """[summary]

    Args:
        logits ([type]): logits, seq_len x batch_size
        Y ([type]): seq_len x batch_size

    Returns:
        [type]: [description]
    """
    batch_size = logits.shape[1]
    Y_pred = (logits >= 0).int()
    mis_match = (Y != Y_pred).int()
    match_sum = (torch.sum(mis_match, dim=0) == 0).int().sum()
    return match_sum

# def test_num_correct_samples():
#     x1 = torch.tensor([[0.5, 0], [-0.5, 3.6], [3.0, 1.5], [-0.2, -2], [1, -0.01]])
#     y1 = torch.tensor([[1, 0], [0, 1], [1, 1], [0, 0], [1, 0]])
#     assert (num_correct_samples(x1, y1) == 1)

#     x2 = torch.tensor([[0.5, 0], [-0.5, 3.6], [3.0, 1.5], [-0.2, -2], [1, -0.01]])
#     y2 = torch.tensor([[1, 1], [0, 1], [1, 1], [0, 0], [1, 0]])
#     assert (num_correct_samples(x2, y2) == 2)

#     x3 = torch.tensor([[0.5, 0], [-0.5, 3.6], [-3.0, 1.5], [-0.2, 2], [1, -0.01]])
#     y3 = torch.tensor([[1, 1], [0, 1], [1, 1], [0, 0], [1, 0]])
#     assert (num_correct_samples(x3, y3) == 0)


def validate(model, val_list):
    criteron = get_criterion()

    with torch.no_grad():
        total_loss = 0
        total_batch = 0
        total_correct = 0
        for i, (X, Y) in enumerate(val_list):
            logit_out = model(X)
            num_correct = num_correct_samples(logit_out, Y)
            loss = criteron(torch.transpose(logit_out, 0, 1),
                            torch.transpose(Y.float(), 0, 1))
            total_loss += loss.item()
            total_batch += 1
            total_correct += num_correct

        avg_loss = total_loss/total_batch
        batch_size = val_list[0][0].shape[1]
        num_batch = len(val_list)
        total_samples = batch_size * num_batch
        percent_correct = total_correct.item()/total_samples

    return avg_loss, percent_correct

# def test_validate():
#     dataloaders = get_dataloaders(20, 10, train_batch_num=1000, val_batch_num=500)
#     train_loader = dataloaders['train']
#     val_list = dataloaders['val']

#     model = RNN(4, 100, 1).to(DEVICE)
#     avg_loss, percent_correct = validate(model, val_list)
#     print(avg_loss >= 0)
#     print(percent_correct <= 1 and percent_correct >= 0)


def scheduler(i):
    return 0.1


def train(model, train_loader, val_loader, logger=None, log_freq=100):

    model.train()

    bce_loss = torch.nn.BCEWithLogitsLoss()
    mse_loss = torch.nn.MSELoss()

    opt_g = torch.optim.SGD(
        [model.V_hh, model.c_h],
        lr=alpha_g, momentum=0.9, nesterov=True)

    opt_y = torch.optim.SGD(
        [model.W_hy, model.b_y],
        lr=alpha_y, momentum=0.9, nesterov=True)
    opt_f = torch.optim.SGD(
        [model.W_xh, model.W_hh, model.b_h],
        lr=alpha_f, momentum=0.9, nesterov=True)
    # opt_f = torch.optim.SGD([model.W_xh, model.W_hh, model.b_h], lr=alpha_f, momentum=0.9, nesterov=True)

    for iteration, (X, Y) in enumerate(train_loader):

        seq_len, batch_sz = Y.size()

        # f_update_prob = (iteration/TRAINING_BATCH_NUM)
        f_update_prob = scheduler(iteration)

        update_f = (torch.rand(1) < f_update_prob)

        if update_f:
            ###### Update (W_hh, W_hx, b_h) and (W_hy, b_y) ##########

            # Forward pass
            # - forward_activations
            # - and local_targets
            total_y_loss = 0
            local_targets_no_grad = []
            forward_activations_local_grad = []
            h_i = torch.zeros(
                [batch_sz, model.hidden_dim],
                dtype=torch.float, device=DEVICE)
            opt_y.zero_grad()  # to accumulate
            for step in range(seq_len):

                # Calculate forward activations
                x_i = X[step, :, :]
                h_i = model.act(x_i @ model.W_xh + h_i.detach()
                                @ model.W_hh + model.b_h)
                forward_activations_local_grad.append(h_i)

                # Calculate local targets and update (W_hh, b_h)
                h_i_detached = h_i.detach().requires_grad_()
                out_logit = h_i_detached @ model.W_hy + model.b_y
                local_loss = bce_loss(
                    torch.transpose(out_logit, 0, 1),
                    Y[step, :].float().unsqueeze_(0))
                local_loss.backward()
                with torch.no_grad():
                    total_y_loss += local_loss
                with torch.no_grad():
                    local_target_i = h_i_detached - alpha_i * h_i_detached.grad
                    local_targets_no_grad.append(local_target_i)

            opt_y.step()

            # Backward pas
            # - calculate combined targets and local_f_loss
            # - update (W_hh, W_xh, b_h)
            past_target_num = 0
            combined_target = None  # keep tack of last combined_target
            opt_f.zero_grad()
            total_f_loss = 0
            for step in range(seq_len-1, -1, -1):
                if step == seq_len-1:
                    with torch.no_grad():
                        combined_target = local_targets_no_grad[step]
                    past_target_num += 1
                else:
                    local_target_weight = 1/(1+past_target_num)
                    local_target = local_targets_no_grad[step]

                    past_target_weight = 1 - local_target_weight
                    # Using DTP
                    with torch.no_grad():
                        past_target = forward_activations_local_grad[step] - model.act(X[step + 1, :, :] @ model.W_xh +
                                                                                       forward_activations_local_grad[step + 1] @ model.V_hh + model.c_h) + model.act(
                            X[step + 1, :, :] @ model.W_xh + combined_target @ model.V_hh + model.c_h)

                        combined_target = local_target_weight * local_target + past_target_weight * past_target

                # Accumulate (W_xh, W_hh, b_h) grad
                local_f_loss = mse_loss(
                    forward_activations_local_grad[step],
                    combined_target)
                local_f_loss.backward()
                with torch.no_grad():
                    total_f_loss += local_f_loss
            opt_f.step()

            ## logging
            if logger:
                logger.log_metric("y_loss", iteration, total_y_loss/seq_len)
                logger.log_metric("f_loss", iteration, total_f_loss/seq_len)

        else:
            ###### Update V_hh and c_h ##########
            h_i = torch.zeros(
                [batch_sz, model.hidden_dim],
                dtype=torch.float, device=DEVICE)
            activations = []

            for step in range(seq_len):
                x_i = X[step, :, :]
                with torch.no_grad():
                    h_i = model.act(x_i @ model.W_xh + h_i @
                                    model.W_hh + model.b_h)
                    activations.append(h_i)

            g_loss_total = 0

            for step in range(seq_len-1, 0, -1):
                x_i = X[step, :, :]
                rec = model.act(
                    x_i @ model.W_xh + activations[step] @ model.V_hh + model.c_h)  # G-function
                g_loss = mse_loss(activations[step-1], rec)
                g_loss_total += g_loss

            opt_g.zero_grad()
            g_loss_total.backward()
            opt_g.step()

            ## logging
            if logger:
                logger.log_metric("g_loss", iteration, g_loss_total/seq_len)

        if (iteration+1) % VAL_INTERVAL == 0:
            model.eval()
            val_loss, percent_correct = validate(model, val_loader)

            print(
                f"iteration {iteration}   val_loss {val_loss:.8f}     percent_correct {percent_correct:.4f}")
            if logger:
                logger.log_metric("val_loss", iteration, val_loss)
                logger.log_metric("accuracy", iteration, percent_correct)


# def test_train():

#     dataloaders = get_dataloaders(20, 10, train_batch_num=20000, val_batch_num=500)
#     train_loader = dataloaders['train']
#     val_loader = dataloaders['val']

#     model = TarSeq(4, 150, 1).to(DEVICE)

#     train(model, train_loader, val_loader)


def train_tarseq():

    parser = argparse.ArgumentParser(description="Data sequence length")
    parser.add_argument("seq_len", help="sequence length", type=int)
    args = parser.parse_args()
    SEQ_LEN = args.seq_len

    dataloaders = get_dataloaders(
        BATCH_SIZE, SEQ_LEN, train_batch_num=TRAINING_BATCH_NUM,
        val_batch_num=VAL_BATCH_NUM)
    train_loader = dataloaders['train']
    val_loader = dataloaders['val']

    model = TarSeq(4, HIDDEN_SIZE, 1).to(DEVICE)

    proj = neptune.init(project_qualified_name="peterpdai/test")
    exp = proj.create_experiment(
        name='tarseq_len_10',
        params={'seq_len': SEQ_LEN, 'alpha_i': alpha_i, 'alpha_f': alpha_f,
                'alpha_g': alpha_g, 'f_prob': f_update_prob},
        upload_source_files=['tarseq_training.py', 'utils.py'],
        tags=["tarseq", "testing"],)

    train(model, train_loader, val_loader, exp, 100)

# # optuna template
# def objective(trial):
#     # code
#     return evaluation_score

# study = optuna.create_study()
# study.optimize(objective, n_trials=num_trials)


def objective(trial):

    # Hyperparameters to search
    seq_len = trial.suggest_int("seq_len", SEQ_LEN, SEQ_LEN)
    lr = trial.suggest_float("lr", 1e-4, 1, log=True)
    momentum = trial.suggest_float("momentum", 0.1, 0.99)

    # Prepare data
    dataloaders = get_dataloaders(
        BATCH_SIZE,
        seq_len,
        train_batch_num=TRAINING_BATCH_NUM,
        val_batch_num=VAL_BATCH_NUM)
    train_loader = dataloaders['train']
    val_list = dataloaders['val']

    rnn = RNN(4, HIDDEN_SIZE, 1).to(DEVICE)

    criterion = get_criterion()
    optimizer = torch.optim.SGD(
        rnn.parameters(),
        lr=lr, momentum=momentum, nesterov=True)

    # Training loop
    best_acc = 0

    for i, (X, Y) in enumerate(train_loader):
        rnn.train()
        optimizer.zero_grad()
        logits = rnn(X)
        loss = criterion(torch.transpose(logits, 0, 1),
                         torch.transpose(Y.float(), 0, 1))
        loss.backward()
        optimizer.step()

        if (i+1) % VAL_INTERVAL == 0:
            rnn.eval()
            val_loss, percent_correct = validate(rnn, val_list)

            if percent_correct > best_acc:
                best_acc = percent_correct

            print(
                f"iteration {i}   train_loss {loss.item():.8f}    val_loss {val_loss:.8f}     percent_correct {percent_correct:.4f}")
            trial.report(percent_correct, i)

            if trial.should_prune():
                raise optuna.exceptions.TrialPruned()

    return best_acc


# def main():

#     # Hyperparameters
#     hparams = {
#         'training_batch_num': 20000,
#         'val_batch_num': 500,
#         'batch_size': 20,
#         'seq_len': 10,
#         'hidden_size': 100, # todo
#         'val_iterval': 100,
#     }

#     # Prepare data
#     dataloaders = get_dataloaders(
#         hparams['batch_size'],
#         hparams['seq_len'],
#         train_batch_num=hparams['training_batch_num'],
#         val_batch_num=hparams['val_batch_num'])
#     train_loader = dataloaders['train']
#     val_list = dataloaders['val']

#     # Define model
#     rnn = RNN(4, hparams['hidden_size'], 1).to(DEVICE)

#     # Init logger
#     proj = neptune.init(project_qualified_name="peterpdai/test")
#     exp_logger = proj.create_experiment(
#         name='sub_string',
#         params=hparams,
#         upload_source_files=['tarseq.py', 'utils.py'],
#         tags=["rnn", "backprop", "sub_string_finder"],
#     )

#     # Main training procedure
#     train(rnn, train_loader, val_list, exp_logger, hparams['val_iterval'])


def hparameter_search():
    parser = argparse.ArgumentParser(
        description="Hyperparameter search for a given sequence length")
    parser.add_argument("seq_len", help="sequence length", type=int)
    args = parser.parse_args()
    SEQ_LEN = args.seq_len

    proj = neptune.init(
        project_qualified_name="peterpdai/tarseq-hparams-search")
    exp = proj.create_experiment(
        name=f'optuna-sweep_length_{SEQ_LEN:02d}',
        upload_source_files=['tarseq_hsearch.py', 'utils.py'],
        tags=["rnn", "hyperparameter-search"])

    neptune_callback = opt_utils.NeptuneCallback(
        log_study=True, log_charts=True)

    study = optuna.create_study(
        direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(objective, n_trials=100, callbacks=[neptune_callback])

    pruned_trials = [t for t in study.trials if t.state ==
                     optuna.trial.TrialState.PRUNED]
    complete_trials = [
        t for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE]

    print("Study statistics: ")
    print("  Number of finished trials: ", len(study.trials))
    print("  Number of pruned trials: ", len(pruned_trials))
    print("  Number of complete trials: ", len(complete_trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))


if __name__ == "__main__":
    train_tarseq()
