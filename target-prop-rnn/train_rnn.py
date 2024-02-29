import argparse
import os
from datetime import datetime
from re import I

import hydra
import neptune
import neptunecontrib.monitoring.optuna as opt_utils
import numpy as np
import optuna
import torch
from attrdict import AttrDict
from omegaconf import DictConfig, OmegaConf
from torch import nn, optim
from torch.nn.functional import one_hot
from torch.nn.modules.loss import BCEWithLogitsLoss

import datasets
import utils

# torch.manual_seed(123)
# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# LOG_INTERVAL = 1
# VAL_INTERVAL = 500
# # N_TRAINING = 500000  # 25000 batches
# N_TRAINING = 1000000 # 50000 batches
# N_VAL = 10000
# N_TEST = 10000

# UPDATE_G_FUNC_ONLY_BEFORE = 2000



class RNN(nn.Module):
    def __init__(self, task, seq_len, method, in_dim, out_dim, device,
                 hidden_dim, actv_fn, batch_size, update_g_before, opt_params):
        super(RNN, self).__init__()

        self.task = task
        self.seq_len = seq_len
        self.method = method
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.device = device
        self.hidden_dim = hidden_dim
        self.actv_fn = actv_fn
        self.batch_size = batch_size
        self.update_g_before = update_g_before
        self.opt_params = opt_params
        if self.actv_fn == 'tanh':
            self.act = nn.Tanh()
        elif self.actv_fn == 'relu':
            self.act = nn.ReLU()
        else:
            print("please set the actv in model")
            exit()

        # learnable parameters
        # orthogonal initialization following original DTP implementation and TPTT
        #
        self.W_hh = nn.Parameter(
            utils.rand_ortho(
                (self.hidden_dim, self.hidden_dim),
                np.sqrt(6. / (self.hidden_dim + self.hidden_dim))))
        self.W_xh = nn.Parameter(torch.empty(self.in_dim, self.hidden_dim))
        nn.init.normal_(self.W_xh, 0, 0.1)
        self.W_hy = nn.Parameter(torch.empty(self.hidden_dim, self.out_dim))
        nn.init.normal_(self.W_hy, 0, 0.1)
        self.b_h = nn.Parameter(torch.zeros([self.hidden_dim]))
        self.b_y = nn.Parameter(torch.zeros([self.out_dim]))

        if self.method == 'backprop':
            self.optimizers = {
                'opt_bp':
                optim.Adam(self.parameters(),
                           lr=self.opt_params.opt_bp.lr,
                           betas=(self.opt_params.opt_bp.beta1,
                                  self.opt_params.opt_bp.beta2))
            }

        if self.method == 'targetprop':
            # additional parameters for inverse function g_fun
            self.V_hh = nn.Parameter(
                utils.rand_ortho(
                    (self.hidden_dim, self.hidden_dim),
                    np.sqrt(6. / (self.hidden_dim + self.hidden_dim))))
            self.c_h = nn.Parameter(torch.zeros([self.hidden_dim]))
            self.optimizers = {
                'opt_f':
                optim.Adam(
                    [self.W_xh, self.W_hh, self.b_h, self.W_hy, self.b_y],
                    lr=self.opt_params.opt_f.lr,
                    betas=(self.opt_params.opt_f.beta1,
                           self.opt_params.opt_f.beta2)),
                'opt_g':
                optim.Adam([self.V_hh, self.c_h],
                           lr=self.opt_params.opt_g.lr,
                           betas=(self.opt_params.opt_g.beta1,
                                  self.opt_params.opt_g.beta2))
            }

        # loss
        self.cross_entropy_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

    def forward(self, x):

        # x of size [batch_size, seq_len] of torch.long
        # conver to [seq_len, batch_size, vocab_size] of float
        x = one_hot(x.transpose(0,
                                1)).float()  # seq_len x batch_sz x vocab_size
        seq_len, batch_size, vocab_size = x.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=self.device)
        out_logits = []
        for step in range(seq_len):
            current_x = x[step, :, :]
            h = self.act(current_x @ self.W_xh + h @ self.W_hh + self.b_h)
            out_logits.append(h @ self.W_hy + self.b_y)
        return torch.stack(out_logits, dim=0).squeeze_().transpose(
            0, 1)  # batch_sz x seq_len x vocab_size

    def f_func(self, h, x):
        # one step forward in time, keep backward gradient of parameters
        return self.act(h @ self.W_hh + x @ self.W_xh + self.b_h)

    def g_func(self, hp1, xp1):
        # given h_{t+1} and x_{t+1}, calculate h_{t}
        # self.W_xh is not trainable in backward step using detach()
        return self.act(hp1 @ self.V_hh + xp1 @ self.W_xh.detach() + self.c_h)

    def y_func(self, h):
        # hidden state to output
        # output are not normalized
        return h @ self.W_hy + self.b_y

    def train_step(self, task, batch_idx, batch_X, batch_Y):
        # process one batch of input and update parameters
        # batch_X: torch.long, batch_size * seq_len
        # batch_Y: torch.long, batch_size * seq_len

        if self.method == 'backprop':
            # impliment backprop
            opt_bp = self.optimizers['opt_bp']
            opt_bp.zero_grad()
            logits = self(batch_X)  # logits, batch_size * seq_len * vocab size
            train_loss = self.cross_entropy_loss(logits.permute(0, 2, 1),
                                                 batch_Y)
            train_loss.backward()
            opt_bp.step()
            return {'train_loss': train_loss}

        if self.method == 'targetprop':
            batch_X = one_hot(batch_X.transpose(0, 1)).float()
            # CORE of the algorithm
            # impliment target prop through time
            opt_g = self.optimizers[
                'opt_g']  # update inverse function: V_hh, c_h
            opt_f = self.optimizers[
                'opt_f']  # update forward functions: W_hh, b_h, W_xh, W_hy, b_y

            batch_size = self.batch_size
            seq_len = batch_Y.size(1)

            if batch_idx <= self.update_g_before:  # update inverse g_func only before this number of steps

                h_t = torch.zeros([batch_size, self.hidden_dim],
                                  dtype=torch.float,
                                  device=self.device)
                g_loss = torch.zeros(1, device=self.device)
                f_loss = torch.zeros(1, device=self.device)
                opt_g.zero_grad()

                for t in range(seq_len):
                    x_t = batch_X[t, :, :]
                    if t == 0:
                        with torch.no_grad():
                            h_prev = self.f_func(h_t, x_t)
                    else:
                        with torch.no_grad():
                            h_t = self.f_func(h_prev, x_t)
                        rec = self.g_func(h_t, x_t)  # trainable, V_hh, c_h
                        g_loss_step = self.mse_loss(h_prev, rec)
                        g_loss += g_loss_step
                        h_prev = h_t
                g_loss.backward()
                opt_g.step()

            else:  # after UPDATE_G_FUNC_ONLY_BEFORE steps, update both f_fun and g_fun in the same step
                lr_i = self.opt_params['lr_i']
                h_t = torch.zeros([batch_size, self.hidden_dim],
                                  dtype=torch.float,
                                  device=self.device)
                Hs = []
                local_targets = []
                g_loss = torch.zeros(1, device=self.device)
                f_loss = torch.zeros(1, device=self.device)
                opt_g.zero_grad()
                opt_f.zero_grad()

                # forward pass to get activations and local targets at each time step
                for t in range(seq_len):
                    x_t = batch_X[t, :, :]
                    h_t = self.f_func(h_t.detach(), x_t)
                    Hs.append(h_t)
                    h_t_ = h_t.detach().clone().requires_grad_(
                    )  # only to compute local target
                    y_pred = self.y_func(
                        h_t_)  # y_pred: batch_size * vocab_size
                    y_loss_step = self.cross_entropy_loss(
                        y_pred, batch_Y[:, t])
                    y_loss_step.backward(
                    )  # accumulate W_hy and b_y grad, and set local target
                    with torch.no_grad():
                        target_t = h_t_ - lr_i * h_t_.grad
                        local_targets.append(target_t)
                    if t > 0:  # reconstruct the previou step hidden state'
                        rec = self.g_func(Hs[-1].detach(), x_t)
                        g_loss += self.mse_loss(rec, Hs[-2].detach())

                # TODO: below is main step of the algorithm:
                # backward pass to accumulate the targets with different target propagation
                # activations are in Hs list, and local targets are in local_targets list, W_hy and b_y grad are
                # accmulated in the variables throughout the forward pass
                combined_target = None
                for t_ in range(seq_len - 1, -1, -1):
                    if t_ == seq_len - 1:
                        combined_target = local_targets[t_]
                        f_loss += self.mse_loss(Hs[t_],
                                                combined_target.detach())

                    else:  # if not last step, update targets with propagated information from later steps
                        # DTP
                        with torch.no_grad(
                        ):  # no need to keep track of grad when calculating targets
                            back_proped_target = self.g_func(
                                combined_target, batch_X[t_ + 1, :, :])
                            linear_correction = Hs[t_] - self.g_func(
                                Hs[t_ + 1], batch_X[t_ + 1, :, :])
                            back_proped_target = back_proped_target + linear_correction
                            combined_target = back_proped_target + (
                                local_targets[t_] - Hs[t_]) / (t_ + 1)
                        f_loss += self.mse_loss(Hs[t_],
                                                combined_target.detach())

                g_loss.backward(retain_graph=True)
                opt_g.step()  # update V_hh, c_h

                f_loss.backward()  # accmulate grad for W_xh, W_hh, b_h
                opt_f.step()  # update: W_xh, W_hh, b_h, and W_hy, b_y

            return {'f_loss': f_loss, 'g_loss': g_loss}

    def validate(self, val_loader):
        # select_idx = torch.randint(0, VAL_BATCH_NUM)
        self.eval()
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            total_batches = 0
            total_correct = 0
            for i, (X, Y) in enumerate(val_loader):
                X, Y = X.to(self.device), Y.to(self.device)
                logit_out = self(
                    X)  # logit_out: batch_size * seq_len * vocab_size
                num_correct = utils.num_correct_samples(
                    logit_out, Y)  # number of correct samples in batch
                loss = self.cross_entropy_loss(logit_out.permute(0, 2, 1), Y)
                total_loss += loss
                total_samples += X.size(0)
                total_batches += 1
                total_correct += num_correct
            avg_loss = total_loss / total_batches
            accuracy = total_correct / total_samples
        self.train()
        return avg_loss, accuracy


def train_with_hparams(cfg: DictConfig, logger=None, optuna_trial=None):
    # cfg.neptune_api_token = 'Hidden'
    print(
        "\033[92m======================= Experiment Hyperparameters ===================="
    )
    # print("\n".join("{}\t{}".format(k, v) for k, v in dict(cfg).items()))
    print(cfg.pretty())
    print(
        "=======================================================================\033[0m"
    )

    train_loader, val_loader = datasets.prepare_loaders(
        cfg.dataset.name, 
        cfg.dataset.seq_len, 
        cfg.batch_size,
        cfg.dataset.n_train, 
        cfg.dataset.n_val)

    model = RNN(task=cfg.dataset.name,
                seq_len=cfg.dataset.seq_len,
                method=cfg.method.name,
                in_dim=cfg.dataset.input_dim,
                out_dim=cfg.dataset.output_dim,
                device=cfg.device,
                hidden_dim=cfg.hidden_dim,
                actv_fn=cfg.actv_fn,
                batch_size=cfg.batch_size,
                update_g_before=cfg.g_only_before_this,
                opt_params=cfg.method)

    model.to(model.device)

    model.train()
    step_results_list = []
    best_accuracy = 0.

    for batch_idx, (batch_X, batch_Y) in enumerate(train_loader):
        batch_X, batch_Y = batch_X.to(model.device), batch_Y.to(model.device)

        step_result = model.train_step(model.task, batch_idx, batch_X, batch_Y)

        # if training loss becomes nan, return immeidately
        if model.method == 'targetprop' and (torch.isnan(
                step_result['g_loss']) or torch.isnan(step_result['f_loss'])):
            return best_accuracy
        if model.method == 'backprop' and (torch.isnan(
                step_result['train_loss'])):
            return best_accuracy
        # if backprop, result is {'train_loss': float}
        # if targetprop, result is {'f_loss': float, 'g_loss': float}
        step_results_list.append(step_result)

        # send loss to neptune logger
        if (batch_idx + 1) % cfg.log_interval == 0:
            if logger:
                if model.method == 'backprop':
                    logger.log_metric('train_loss', batch_idx,
                                      step_result['train_loss'])
                if model.method == 'targetprop':
                    logger.log_metric('f_loss', batch_idx,
                                      step_result['f_loss'])
                    logger.log_metric('g_loss', batch_idx,
                                      step_result['g_loss'])

        if batch_idx == 0 or (batch_idx + 1) % cfg.val_interval == 0:
            val_loss, val_acc = model.validate(val_loader)
            if torch.isnan(val_loss) or torch.isnan(val_acc):
                return best_accuracy
            # val_loss = torch.tensor([0.])
            # val_acc = torch.tensor([0.])
            print_train_stats = f'val_loss: {val_loss.item():.8f}\t val_acc: {val_acc.item():.4f}'

            if logger:
                logger.log_metric('val_loss', batch_idx, val_loss)
                logger.log_metric('val_acc', batch_idx, val_acc)

            if model.method == 'backprop':
                avg_train_loss = torch.tensor(
                    [loss['train_loss'] for loss in step_results_list]).mean()
                step_results_list = []
                print(
                    f'Step {batch_idx:5d}:\t avg_train_loss: {avg_train_loss.item():.8f}\t\t'
                    + print_train_stats)

            if model.method == 'targetprop':
                avg_f_loss = torch.tensor(
                    [loss['f_loss'] for loss in step_results_list]).mean()
                avg_g_loss = torch.tensor(
                    [loss['g_loss'] for loss in step_results_list]).mean()
                # could be done in 1 step
                step_results_list = []
                print(
                    f'Step {batch_idx:5d}:\t avg_f_loss: {avg_f_loss.item():.8f}\t avg_g_loss: {avg_g_loss.item():.8f}\t\t'
                    + print_train_stats)
            if val_acc > best_accuracy:
                best_accuracy = val_acc
                torch.save(model, 'best' + str(batch_idx) + '.model')
            if optuna_trial:  # used for hyperparameter truning
                optuna_trial.report(val_acc, batch_idx)
                if val_acc > best_accuracy:
                    best_accuracy = val_acc
                    torch.save(model, 'best' + str(batch_idx) + '.model')
                if optuna_trial.should_prune():
                    raise optuna.exceptions.TrialPruned()
    if optuna_trial:
        return best_accuracy
    torch.save(model, 'best' + str(batch_idx) + '.model')


@hydra.main(config_path='conf', config_name="training_config")
def run_training(cfg: DictConfig):
    if cfg.use_neptune_logger:
        neptune.init(project_qualified_name=cfg.neptune_project)
        logger = neptune.create_experiment(
            name=cfg.neptune_exp.name,
            upload_source_files=cfg.neptune_exp.upload_files,
            # upload_source_files=['train_rnn.py'],
            params=dict(cfg))
        # logger.log_artifact(cfg.neptune_exp.artifacts)
    else:
        logger = None
    train_with_hparams(cfg, logger)


if __name__ == "__main__":
    run_training()
