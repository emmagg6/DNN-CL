
from numpy.lib.financial import nper
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
from torch.nn.functional import linear, one_hot
from tempOrder import TempOrderTask


import utils


# torch.manual_seed(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAINING_BATCH_NUM = 100000
VAL_BATCH_NUM = 500
BATCH_SIZE = 20
HIDDEN_SIZE = 100

VAL_INTERVAL = 500
LOG_INVERVAL = 1

bp_alpha = 0.0001
tp_alpha_i = 1
tp_alpha_f = 0.01
tp_alpha_g = 0.1
momentum=0.9

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
        seq_len, batch_size, vocab_size = x.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        for step in range(seq_len):
            current_x = x[step, :, :]
            h = self.act(torch.mm(current_x, self.W_xh) + torch.mm(h, self.W_hh) + self.b_h)
        out_logits = torch.mm(h, self.W_hy) + self.b_y
        return out_logits

def batch_correct(out_logits, Y):
    out_logits = torch.tensor(out_logits, device=DEVICE)
    Y = torch.tensor(Y, device=DEVICE)
    # number of correct predicted samples per batch
    idx = torch.argmax(out_logits, dim=1)
    return sum(Y.gather(1, idx.view(-1, 1))) # select entry based on prediction, 1 if correct

def validate(model, val_loader, seq_len):
    criterion = torch.nn.CrossEntropyLoss()
    with torch.no_grad():
        total_loss = 0
        total_samples = 0
        total_batches = 0
        total_correct = 0
        for i in range(VAL_BATCH_NUM):
            X, Y = val_loader.generate(BATCH_SIZE, seq_len)
            X = torch.tensor(X, device=DEVICE)
            Y = torch.tensor(Y, device=DEVICE)
            # if i == 0:
            #     print(X[:,0,:])
            #     print(Y[i,:])
            logit_out = model(X) # batch size x 6
            num_correct = batch_correct(logit_out, Y)
            loss = criterion(logit_out, torch.argmax(Y, dim=1))
            total_loss += loss
            total_samples += X.size(1)
            total_batches += 1
            total_correct += num_correct
        avg_loss = total_loss/total_batches
        percent_correct = total_correct/total_samples
    return avg_loss, percent_correct


def bp_train():

    parser = argparse.ArgumentParser(description="Training a simple RNN model")
    parser.add_argument('--T', help='sequence length',
                        type=int, required=True)
    args = parser.parse_args()
    seq_len = args.T


    train_generator = TempOrderTask(np.random.RandomState(123), 'float32')

    model = RNN(6, HIDDEN_SIZE, 4).to(DEVICE)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=bp_alpha, momentum=momentum, nesterov=True)

    neptune.init(project_qualified_name="peterpdai/test")
    neptune.create_experiment(
        name=f'backprop_temporder_seq_len_{seq_len:02d}',
        upload_source_files=['*.py'],
        tags=["backprop", "training", 'tempOrder'])

    for i in range(TRAINING_BATCH_NUM):
        optimizer.zero_grad()

        X, Y = train_generator.generate(BATCH_SIZE, seq_len)
        X = torch.tensor(X, device=DEVICE)
        Y = torch.tensor(Y, device=DEVICE)

        logits = model(X) # batch_size x vocab_size
        assert (logits.shape == torch.Size([BATCH_SIZE, 4]))
        loss = criterion(logits, torch.argmax(Y, dim=1))
        loss.backward()
        optimizer.step()

        if (i+1) % LOG_INVERVAL == 0:
            neptune.log_metric("training_loss", i, loss)

        if (i+1) % VAL_INTERVAL == 0:
            # fixed dataset for each validation 
            model.eval()
            val_generator = TempOrderTask(np.random.RandomState(456), 'float32')
            val_loss, percent_correct = validate(model, val_generator,seq_len)
            print(f"iteration {i+1}   train_loss {loss.item():.4f}    val_loss {val_loss.item():.4f}     \
                    percent_correct {percent_correct.item():.4f}")
            neptune.log_metric("validation_loss", i, val_loss)
            neptune.log_metric("validation_accuracy", i, percent_correct)
            model.train()

def tptt_train():
    parser = argparse.ArgumentParser(description="Training a simple RNN model")
    parser.add_argument('--T', help='sequence length',
                        type=int, required=True)
    args = parser.parse_args()
    seq_len = args.T


    train_generator = TempOrderTask(np.random.RandomState(123), 'float32')

    model = RNN(6, HIDDEN_SIZE, 4).to(DEVICE)
    model.train()

    criterion = torch.nn.CrossEntropyLoss()
    mse_loss = torch.nn.MSELoss()
    opt_g = torch.optim.SGD([model.V_hh, model.c_h], lr=tp_alpha_g, momentum=momentum, nesterov=True)
    opt_f = torch.optim.SGD([model.W_xh, model.W_hh, model.b_h, model.W_hy, model.b_y], lr=tp_alpha_f, momentum=momentum, nesterov=True)
    
    neptune.init(project_qualified_name="peterpdai/test")
    neptune.create_experiment(
        name=f'tptt_temporder_seq_len_{seq_len:02d}',
        upload_source_files=['*.py'],
        tags=["tptt", "training", 'tempOrder'])


    for i in range(TRAINING_BATCH_NUM):
        X, Y = train_generator.generate(BATCH_SIZE, seq_len)
        X = torch.tensor(X, device=DEVICE)
        Y = torch.tensor(Y, device=DEVICE)

        seq_len, batch_sz, _ = X.size()
       
        # forward pass
        h = torch.zeros([batch_sz, model.hidden_dim], dtype=torch.float, device=DEVICE)
        Hs = []
        for t in range(seq_len):
            x_t = X[t,:,:] # batch x 6
            h = model.act(h.detach().clone() @ model.W_hh + x_t @ model.W_xh + model.b_h)
            Hs.append(h)
        # print(h.grad_fn) # tanhback
        h_T = Hs[-1].detach().clone().requires_grad_()
        # print(h_T.grad_fn) # None
        # print(h_T.requires_grad) # True
        out = h_T @ model.W_hy + model.b_y
        y_loss = criterion(out, torch.argmax(Y, dim=1))
        
        opt_f.zero_grad()
        opt_g.zero_grad()

        y_loss.backward()
        ## W_hy, b_y grad still with tensor 
        h_hat_t = (h_T - tp_alpha_i * h_T.grad).detach() # last target
        target_dist = - tp_alpha_i * h_T.grad
        neptune.log_metric("last target dist", i, torch.linalg.norm(torch.mean(target_dist, dim=0)))
        f_loss = 0
        g_loss = 0
        dist_1_total = 0
        dist_2_total = 0
        dist_3_total = 0
        dist_4_total = 0
        for t in range(seq_len-1, -1, -1):
            if t == seq_len-1: # only update f    
                f_loss_t = mse_loss(Hs[t], h_hat_t)
                f_loss += f_loss_t 
            else: # both update f and g
                # accumulate g_loss, reconstruction loss
                rec = model.act(Hs[t+1].detach() @ model.V_hh + X[t+1,:,:] @ model.W_xh.detach() + model.c_h) 
                g_loss_t = mse_loss(rec, Hs[t].detach())
                g_loss += g_loss_t 

                # set target with dtp
                with torch.no_grad():
                    back_proj = model.act(h_hat_t.detach() @ model.V_hh + X[t+1,:,:] @ model.W_xh + model.c_h)

                    dist_2_total += torch.linalg.norm(torch.mean(back_proj - rec, dim=0)) 
                    linear_correction = (Hs[t] - rec)
                    dist_1_total += torch.linalg.norm(torch.mean(Hs[t] - rec, dim=0)) 
                    h_hat_t = back_proj + linear_correction
                    dist_3_total += torch.linalg.norm(torch.mean(h_hat_t - back_proj, dim=0)) 
                f_loss_t = mse_loss(Hs[t], h_hat_t)
                dist_4_total += torch.linalg.norm(torch.mean(h_hat_t - Hs[t], dim=0))
                f_loss += f_loss_t 
        
        neptune.log_metric("dist_1", i, dist_1_total/(seq_len-1))
        neptune.log_metric("dist_2", i, dist_2_total/(seq_len-1))
        neptune.log_metric("dist_3", i, dist_3_total/(seq_len-1))
        neptune.log_metric("dist_4", i, dist_4_total/(seq_len-1))
        
        if i < 1000:
            g_loss.backward()
            opt_g.step()
        else:
            g_loss.backward(retain_graph=True)
            f_loss.backward()
            opt_g.step()
            opt_f.step()            
        

        if (i+1) % LOG_INVERVAL == 0:
            neptune.log_metric("f_loss", i, f_loss)
            neptune.log_metric("g_loss", i, g_loss)
        if (i+1) % VAL_INTERVAL == 0:
            # fixed dataset for each validation 
            # model.eval()
            val_generator = TempOrderTask(np.random.RandomState(456), 'float32')
            val_loss, percent_correct = validate(model, val_generator,seq_len)
            print(f"iteration {i+1}         val_loss {val_loss.item():.4f}     \
                    percent_correct {percent_correct.item():.4f}")
            neptune.log_metric("validation_loss", i, val_loss)
            neptune.log_metric("validation_accuracy", i, percent_correct)
            # model.train()

                



# def tp_train(model, train_loader, val_loader, logger=None):
#     model.train()
#     cross_entropy_loss = torch.nn.CrossEntropyLoss()
#     mse_loss = torch.nn.MSELoss()
#     opt_g = torch.optim.SGD([model.V_hh, model.c_h], lr=lr_g, momentum=momentum, nesterov=True)
#     opt_y = torch.optim.SGD([model.W_hy, model.b_y], lr=lr_y, momentum=momentum, nesterov=True)
#     opt_f = torch.optim.SGD([model.W_xh, model.W_hh, model.b_h], lr=lr_f, momentum=momentum, nesterov=True)

#     for iteration, (X, Y) in enumerate(train_loader):
#         X = one_hot(X.long().transpose(0, 1), num_classes=VOCAB_SIZE).float()  # seq_len x batch_size x vocab_size
#         Y = Y.long()
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
#             if logger:
#                 # print(g_loss_total/seq_len)
#                 logger.log_metric("g_loss", iteration, g_loss_total/seq_len)

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
#             if logger:
#                 logger.log_metric("y_loss", iteration, y_loss_total/seq_len)

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
#             # combined_targets = [] 
#             prop_targets = []
#             for step in range(seq_len-1, -1, -1):
#                 if step == seq_len -1:
#                     prop_targets.append([local_targets[step]])
#                 else:
#                     with torch.no_grad():
#                         temp_targets = []
#                         linear_correction = pre_activations[step] - (X[step+1] @ model.W_xh + model.act(pre_activations[step+1]) @ model.V_hh + model.c_h)
#                         # linear_correction = 0
#                         for target in prop_targets[-1]: # targets from last step
#                             back_projected_targets = X[step+1, :, :] @ model.W_xh + model.act(target) @ model.V_hh + model.c_h
#                             temp_targets.append(back_projected_targets+linear_correction)
#                         temp_targets.append(local_targets[step]) # growing the list by 1
#                         prop_targets.append(temp_targets) # append to the list

#             ## STEP 3: 
#             ## - Update local weights (W_xh, W_hh, b_h) to get closer to dtp_targets
#             f_loss_total = 0
#             h_step = torch.zeros([batch_sz, model.hidden_dim], dtype=torch.float, device=DEVICE)
#             opt_f.zero_grad()
#             for step in range(seq_len):
#                 h_step = X[step, :, :] @ model.W_xh + model.act(h_step).detach() @ model.W_hh + model.b_h 
#                 current_targets = prop_targets.pop()
#                 assert (len(current_targets) == seq_len-step)
#                 for i, target in enumerate(current_targets):
#                     # loss = (1/(seq_len-i))*mse_loss(h_step, target)
#                     loss = mse_loss(h_step, target)
#                     if (step==5) and (i == 0):
#                         logger.log_metric("step_loss", iteration, loss)
#                     f_loss_total += loss
#             f_loss_total.backward()
#             opt_f.step()
#             if logger:
#                 logger.log_metric("f_loss", iteration, f_loss_total/seq_len)
#                 logger.log_metric("f_loss_mean", iteration, (f_loss_total/((seq_len+1)*seq_len/2)))


#         if (iteration+1) % VAL_INTERVAL == 0:
#             model.eval()
#             val_loss, percent_correct = validate(model, val_loader)

#             print(f"iteration {iteration}   val_loss {val_loss:.8f}     percent_correct {percent_correct:.4f}")
#             if logger:
#                 logger.log_metric("val_loss", iteration, val_loss)
#                 logger.log_metric("accuracy", iteration, percent_correct)
#             model.train()

# def run_training():
#     parser = argparse.ArgumentParser(description="Training a simple RNN model with target propagation")
#     parser.add_argument('--seq_len', help='sequence length',
#                         type=int, required=True)
#     args = parser.parse_args()
#     seq_len = args.seq_len

#     train_loader, val_loader = utils.prepare_double_char_datasets(
#         VOCAB_SIZE,
#         seq_len,
#         COPY_NUM,
#         BATCH_SIZE,
#         TRAINING_BATCH_NUM * BATCH_SIZE,
#         VAL_BATCH_NUM * BATCH_SIZE,
#         DEVICE)

#     model = RNN(VOCAB_SIZE, HIDDEN_SIZE, VOCAB_SIZE).to(DEVICE)
#     model.train()

#     neptune.init(project_qualified_name="peterpdai/test")
#     exp = neptune.create_experiment(
#         name=f'target_prop_best_hparams_seq_len_{seq_len:02d}',
#         upload_source_files=['dup_char_tarseq_new.py', 'utils.py'],
#         tags=["target-prop", "training", 'dup_char'])

#     tp_train(model, train_loader, val_loader, logger=exp)


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
    bp_train()
    # tptt_train()
    # run_hparameter_search()
