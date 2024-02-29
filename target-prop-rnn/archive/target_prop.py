
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
from torch.nn.functional import one_hot

import utils
from datasets import expand_seq_dataset


# torch.manual_seed(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAINING_BATCH_NUM = 50000
VAL_BATCH_NUM = 500
BATCH_SIZE = 20
VOCAB_SIZE = 5
COPY_NUM = 2
NOISE_LEVEL = 0.0

HIDDEN_SIZE = 200

VAL_INTERVAL = 500
LOG_INVERVAL = 1


lr_bp = 0.001
lr_i = 0.08131466391415924
lr_f = 3.340630057129555e-05
lr_y = 0.08074230707266497
lr_g = 0.004747009644720251
momentum = 0.9693960711546346


class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super(RNN, self).__init__()
        self.act = torch.nn.Tanh()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        self.W_hh = nn.Parameter(utils.rand_ortho((self.hidden_dim, self.hidden_dim),
                                                  np.sqrt(6. / (self.hidden_dim + self.hidden_dim))))
        self.V_hh = nn.Parameter(utils.rand_ortho((self.hidden_dim, self.hidden_dim),
                                                  np.sqrt(6. / (self.hidden_dim + self.hidden_dim))))
        self.W_xh = nn.Parameter(torch.empty(self.in_dim, self.hidden_dim))
        torch.nn.init.normal_(self.W_xh, 0, 0.1)
        self.W_hy = nn.Parameter(torch.empty(self.hidden_dim, self.out_dim))
        torch.nn.init.normal_(self.W_hy, 0, 0.1)
        self.b_h = nn.Parameter(torch.zeros([self.hidden_dim]))
        self.c_h = nn.Parameter(torch.zeros([self.hidden_dim]))
        self.b_y = nn.Parameter(torch.zeros([self.out_dim]))

    def forward(self, x):
        x = one_hot(x.transpose(0, 1)).float()
        seq_len, batch_size, vocab_size = x.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        out = []
        for step in range(seq_len):
            current_x = x[step, :, :]
            # print(current_x.device)
            h = self.act(current_x @ self.W_xh + h @ self.W_hh + self.b_h)
            out.append(h @ self.W_hy + self.b_y)
        return torch.stack(out, dim=0).squeeze_()


def validate(model, val_loader):
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        total_batches = 0
        total_correct = 0
        for (X, Y) in val_loader:
            logit_out = model(X) 
            # logit_out: seq_len x batch_sz x vocab_size
            # Y: batch_sz x seq_len 
            num_correct = utils.num_correct_samples(logit_out, Y)
            loss = criterion(logit_out.permute(1, 2, 0), Y)
            total_loss += loss
            total_samples += X.size(0)
            total_batches += 1
            total_correct += num_correct
        avg_loss = total_loss/total_batches
        percent_correct = total_correct/total_samples
    return avg_loss, percent_correct





def train(model, train_loader, val_loader, method="backprop", logger=None):

    if (method == "backprop"):
        ce_loss = torch.nn.CrossEntropyLoss()
        mse_loss = torch.nn.MSELoss() 
        opt = torch.optim.SGD(model.parameters(), lr=lr_bp, momentum=momentum, nesterov=True)
        
        for iteration, (X, Y) in enumerate(train_loader):

            out = model(X)
            opt.zero_grad()
            loss = ce_loss(out.permute(1, 2, 0), Y)
            loss.backward()
            opt.step()

            if ((iteration+1) % LOG_INVERVAL == 0) and logger:
                logger.log_metric("train_loss", iteration, loss)

            if (iteration+1) % VAL_INTERVAL == 0:
                val_loss, acc = validate(model, val_loader)
                print(f"iteration: {iteration}       val_loss: {val_loss:.4f}     accuracy: {acc:.4f}")
                if logger:
                    logger.log_metric("val_loss", iteration, val_loss)
                    logger.log_metric("accuracy", iteration, acc)
                model.train()
    
    elif (method == "targetprop"):

        ce_loss = torch.nn.CrossEntropyLoss()
        mse_loss = torch.nn.MSELoss() 
        opt_g = torch.optim.SGD([model.V_hh, model.c_h], lr=lr_g, momentum=momentum, nesterov=True)
        opt_f = torch.optim.SGD([model.W_xh, model.W_hh, model.b_h, model.W_hy, model.b_y], lr=lr_f, momentum=momentum, nesterov=True)

        for iteration, (X, Y) in enumerate(train_loader):
            X = one_hot(X.transpose(0, 1)).float()
            batch_sz, seq_len = Y.size()

            # update inverse functions first before upating forward functions
            ####### Update (W_xh, V_hh and c_h) ##########
            h_i = torch.zeros([batch_sz, model.hidden_dim], dtype=torch.float, device=DEVICE)
            activations = [] # with each step decoupled with .detach()
            for t in range(seq_len):
                x_t = X[t, :, :]
                h_i = x_t.detach() @ model.W_xh + model.act(h_i).detach() @ model.W_hh + model.b_h # pre-activation 
                activations.append(h_i)
            g_loss_total = 0
            opt_g.zero_grad()

            for t in range(seq_len-1):
                with torch.no_grad():
                    h_t_corrupted = (activations[t] + NOISE_LEVEL * torch.randn_like(activations[t]))
                    encoding = model.act(h_t_corrupted) @ model.W_hh + X[t, :, :] @ model.W_xh + model.b_h 
                rec = model.act(encoding.detach()) @ model.V_hh + X[t+1, :,:] @ model.W_xh.detach() + model.c_h

                rec_loss = mse_loss(rec, activations[t].detach())
                g_loss_total += rec_loss 
            
            g_loss_total.backward()
            opt_g.step()
            if logger:
                logger.log_metric("g_loss", iteration, g_loss_total/seq_len)

            
            if iteration >= 0:
                # updating forward functions together after some iterations updating only inverse functions 
                opt_f.zero_grad()
                local_targets = []
                total_f_loss = 0
                for t in range(seq_len):
                    h_t = activations[t].detach().requires_grad_()
                    out = model.act(h_t) @ model.W_hy + model.b_y  # batch x vocab
                    step_loss = ce_loss(out, Y[:,t]) 
                    step_loss.backward() # accumulating W_hy and b_y grad
                    target_t = h_t - lr_i * h_t.grad 
                    local_targets.append(target_t.detach())
                # propagating targets backward 
                propagating_target = local_targets[-1]
                for t in range(seq_len-1, -1, -1): # backward in time
                    if t == seq_len - 1: # directly use local target
                        f_loss_step = mse_loss(activations[t], propagating_target)
                        total_f_loss += f_loss_step 
                    else:
                        # DTP
                        with torch.no_grad():
                            tp_inverse = model.act(propagating_target) @ model.V_hh + X[t+1, :, :] @ model.W_xh + model.c_h
                            linear_correction = activations[t] \
                                                    - (model.act(activations[t+1]) @ model.V_hh \
                                                    + X[t+1, :, :] @ model.W_xh + model.c_h)
                            # linear_correction = 0 # TODO 
                            logger.log_metric("tp_inverse_norm", torch.norm(tp_inverse))
                            logger.log_metric("linear_correction_norm", torch.norm(linear_correction))
                            logger.log_metric("local target step", torch.norm((local_targets[t] - activations[t])))
                            logger.log_metric(" backprojection to activation", torch.norm((tp_inverse - activations[t])))
                            propagating_target = tp_inverse + linear_correction + (local_targets[t] - activations[t])
                        f_loss_step = mse_loss(activations[t], propagating_target)
                        total_f_loss += f_loss_step 
                total_f_loss *= 1 # TODO 
                total_f_loss.backward()
                opt_f.step()
                if logger:
                    logger.log_metric("f_loss_total", iteration, total_f_loss)

            if (iteration+1) % VAL_INTERVAL == 0:
                val_loss, acc = validate(model, val_loader)
                print(f"iteration: {iteration}       val_loss: {val_loss:.4f}     accuracy: {acc:.4f}")
                if logger:
                    logger.log_metric("val_loss", iteration, val_loss)
                    logger.log_metric("accuracy", iteration, acc)

    else:
        print("Please specify the training method: [\"backprop\" | \"targetprop\"]")


def run_training():
    parser = argparse.ArgumentParser(description="Training a simple RNN model with target_prop or backprop")
    parser.add_argument('--seq_len', help='sequence length',
                        type=int, required=True)

    parser.add_argument('--method', help='optimization method', required=True)

    args = parser.parse_args()
    seq_len = args.seq_len
    method = args.method 

    train_loader, val_loader = expand_seq_dataset(
        VOCAB_SIZE,
        seq_len,
        COPY_NUM,
        BATCH_SIZE,
        TRAINING_BATCH_NUM * BATCH_SIZE,
        VAL_BATCH_NUM * BATCH_SIZE,
        DEVICE)


    model = RNN(VOCAB_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(DEVICE)

    project = neptune.init(project_qualified_name="peterpdai/test")
    logger = project.create_experiment(
        name=f'training_rnn_{seq_len:02d}',
        upload_source_files=['target_prop.py', 'utils.py'],
        tags=[method, "training", 'expand_sequence_task'])
    # logger = None

    train(model, train_loader, val_loader, method, logger)



# def objective(trial):

#     ##### Hyperparameters #####
#     # seq_len = trial.suggest_int("seq_len", SEQ_LEN, SEQ_LEN)
#     lr_i = trial.suggest_float("lr_i", 1e-5, 1, log=True)
#     lr_f = trial.suggest_float("lr_f", 1e-5, 1, log=True)
#     lr_y = trial.suggest_float("lr_y", 1e-5, 1, log=True)
#     lr_g = trial.suggest_float("lr_g", 1e-5, 1, log=True)

#     momentum = trial.suggest_float("momentum", 0.0, 0.99)
#     seq_len = 10
    
#     print("lr_i: ", lr_i, "lr_f: ", lr_f, "lr_y: ", lr_y, "lr_g: ", lr_g, "momentum: ", momentum)

#     #### Training setup ####
#     train_loader, val_loader = utils.prepare_double_char_datasets(
#             VOCAB_SIZE,
#             seq_len,
#             COPY_NUM,
#             BATCH_SIZE,
#             TRAINING_BATCH_NUM * BATCH_SIZE,
#             VAL_BATCH_NUM * BATCH_SIZE,
#             DEVICE)
    
#     model = RNN(VOCAB_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(DEVICE)

#     cross_entropy_loss = torch.nn.CrossEntropyLoss()
#     mse_loss = torch.nn.MSELoss()
#     opt_g = torch.optim.SGD([model.W_xh, model.V_hh, model.c_h], lr=lr_g, momentum=momentum, nesterov=True)
#     opt_y = torch.optim.SGD([model.W_hy, model.b_y], lr=lr_y, momentum=momentum, nesterov=True)
#     opt_f = torch.optim.SGD([model.W_xh, model.W_hh, model.b_h], lr=lr_f, momentum=momentum, nesterov=True)


#     #### Training procedure ####
#     best_accuracy = 0.
#     for iteration, (X, Y) in enumerate(train_loader):
#         X = one_hot(X.transpose(0, 1)).float()  # seq_len x batch_size x vocab_size
#         batch_sz, seq_len = Y.size()
#         update_g, update_y, update_f = scheduler3(iteration)

#         if update_g:
#             ###### Update (W_xh, V_hh and c_h) ##########
#             h_i = torch.zeros([batch_sz, model.hidden_dim], dtype=torch.float, device=DEVICE)
#             pre_activations_no_grad = []
#             for step in range(seq_len):
#                 x_i = X[step, :, :]
#                 with torch.no_grad():
#                     h_i = x_i @ model.W_xh + model.act(h_i) @ model.W_hh + model.b_h # pre-activation 
#                     pre_activations_no_grad.append(h_i)
#             g_loss_total = 0
#             opt_g.zero_grad()
#             for step in range(seq_len-1, 0, -1):
#                 x_i = X[step, :, :]
#                 rec = x_i @ model.W_xh.detach() + model.act(pre_activations_no_grad[step]) @ model.V_hh + model.c_h  # G-function
#                 g_loss_step = mse_loss(rec, pre_activations_no_grad[step-1])
#                 g_loss_total += g_loss_step
#             g_loss_total.backward()
#             opt_g.step()

#         if update_y:
#             ###### Update (W_hy, b_y) ##########
#             y_loss_total = 0
#             a_i = torch.zeros([batch_sz, model.hidden_dim], dtype=torch.float, device=DEVICE)
#             opt_y.zero_grad()
#             for step in range(seq_len):
#                 x_i = X[step, :, :]
#                 with torch.no_grad():
#                     a_i = model.act(x_i @ model.W_xh + a_i @ model.W_hh + model.b_h)
#                 out_logit = a_i @ model.W_hy + model.b_y
#                 y_loss_step = cross_entropy_loss(out_logit, Y[:, step])
#                 y_loss_total += y_loss_step
#             y_loss_total.backward()
#             opt_y.step()

#         if update_f: 
#             ###### Update (W_hh, W_hx, b_h) ##########

#             ## STEP 1: 
#             ## activations: Forward activations
#             ## local_targets: Local targets  
#             local_targets = []
#             pre_activations = []
#             h_i = torch.zeros([batch_sz, model.hidden_dim], dtype=torch.float, device=DEVICE)
#             opt_y.zero_grad() # grad for W_hy and b_y not used
#             for step in range(seq_len):
#                 x_i = X[step, :, :]
#                 with torch.no_grad():
#                     h_i = x_i @ model.W_xh + model.act(h_i) @ model.W_hh + model.b_h
#                     pre_activations.append(h_i)
#                 h_i_detached = h_i.detach().requires_grad_()
#                 out_logit = model.act(h_i_detached) @ model.W_hy + model.b_y
#                 local_loss = cross_entropy_loss(out_logit, Y[:, step])
#                 local_loss.backward()
#                 with torch.no_grad():
#                     local_target_i = h_i_detached - lr_i * h_i_detached.grad
#                     local_targets.append(local_target_i) 
                    
#             ## STEP 2: 
#             ## - USE DTP to set recurrent targets 
#             combined_targets = [] 
#             for step in range(seq_len-1, -1, -1):
#                 if step == seq_len -1:
#                     combined_targets.append(local_targets[step]/(step+1))
#                 else:
#                     with torch.no_grad():
#                         local_target_weight = 1.0/(step+1)
#                         back_projected_targets = X[step+1, :, :] @ model.W_xh + model.act(combined_targets[-1]) @ model.V_hh + model.c_h
#                         linear_correction = pre_activations[step] - (X[step+1] @ model.W_xh + model.act(pre_activations[step+1]) @ model.V_hh + model.c_h)
#                         local_combined_target = (back_projected_targets + linear_correction) + local_target_weight * local_targets[step] # pre-activation
#                         combined_targets.append(local_combined_target)

#             ## STEP 3: 
#             ## - Update local weights (W_xh, W_hh, b_h) to get closer to dtp_targets
#             f_loss_total = 0
#             h_step = torch.zeros([batch_sz, model.hidden_dim], dtype=torch.float, device=DEVICE)
#             opt_f.zero_grad()
#             for step in range(seq_len):
#                 h_step = X[step, :, :] @ model.W_xh + model.act(h_step).detach() @ model.W_hh + model.b_h
#                 f_loss_step = mse_loss(h_step, combined_targets[-1-step])
#                 f_loss_total += f_loss_step
#             f_loss_total.backward()
#             opt_f.step()

#         if (iteration+1) % VAL_INTERVAL == 0:
#             model.eval()
#             val_loss, percent_correct = validate(model, val_loader)

#             print(f"iteration {iteration}   val_loss {val_loss:.8f}     percent_correct {percent_correct:.4f}")
#             if torch.isnan(val_loss):
#                 model.train()
#                 return best_accuracy 
#             if percent_correct > best_accuracy:
#                 best_accuracy = percent_correct 
 
#             trial.report(percent_correct, iteration)
#             if trial.should_prune():
#                 raise optuna.exceptions.TrialPruned()
#     return best_accuracy


# def run_hparameter_search():
#     # parser = argparse.ArgumentParser(description="Hyperparameter search for a given sequence length")
#     # parser.add_argument("seq_len", help="sequence length", type=int)
#     # args = parser.parse_args()
#     # SEQ_LEN = args.seq_len
#     seq_len = 10

#     neptune.init(project_qualified_name="peterpdai/tarseq-hparams-search")
#     neptune.create_experiment(name=f'target-prop_length_{seq_len:02d}',
#                                  upload_source_files=['dup_char_tarseq_new.py', 'utils.py'],
#                                  tags=["target-prop", "hyperparameter-search"])

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
#     # run_hparameter_search()
