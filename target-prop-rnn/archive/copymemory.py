
from re import I
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
import datasets


# torch.manual_seed(0)
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

TRAINING_BATCH_NUM = 100000
VAL_BATCH_NUM = 500

VAL_INTERVAL = 500
LOG_INVERVAL = 1


class RNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, actv):
        super(RNN, self).__init__()

        # model architecture
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim
        if actv == 'tanh':
            self.act = torch.nn.Tanh()
        elif actv == 'relu':
            self.act = torch.nn.ReLU()
        else:
            print("please set the actv in model")
            exit()

        # learnable parameters
        self.W_hh = nn.Parameter(utils.rand_ortho((self.hidden_dim, self.hidden_dim),
                np.sqrt(6. / (self.hidden_dim + self.hidden_dim))))
        self.W_xh = nn.Parameter(torch.empty(self.in_dim, self.hidden_dim))
        torch.nn.init.normal_(self.W_xh, 0, 0.1)
        self.W_hy = nn.Parameter(torch.empty(self.hidden_dim, self.out_dim))
        torch.nn.init.normal_(self.W_hy, 0, 0.1)
        self.b_h = nn.Parameter(torch.zeros([self.hidden_dim]))
        self.b_y = nn.Parameter(torch.zeros([self.out_dim]))

        # loss 
        self.cross_entropy_loss = torch.nn.CrossEntropyLoss()
        self.mse_loss = torch.nn.MSELoss()
        
        

    def forward(self, x):

        # x of size [batch_size, seq_len]
        # conver to [seq_len, batch_size, vocab_size]

        x = one_hot(x.transpose(0, 1)).float() # seq_len x batch_sz x vocab_size 
        seq_len, batch_size, vocab_size = x.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=DEVICE)
        out_logits = []
        for step in range(seq_len):
            current_x = x[step, :, :]
            h = self.act(current_x @ self.W_xh + h @ self.W_hh + self.b_h)
            out_logits.append(h @ self.W_hy + self.b_y)  
        return torch.stack(out_logits, dim=0).squeeze_() # seq_len x batch_sz x vocab_size 

    def set_method(self, method):
        self.method = method

    def set_optimizer(self, optimizer_type='adam', learning_rate=0.001):
        if self.method == 'backprop':
            if optimizer_type == "adam":
                self.optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
            else:
                print("Please specify the correcct optimizer type")
                exit()
        elif self.method == 'targetprop':
            if optimizer_type == "adam":
                self.optimizer = {
                    'opt_f': torch.optim.Adam(self.parameters(), lr=learning_rate['lr_f']),
                    'opt_g': torch.optim.Adam(self.parameters(), lr=learning_rate['lr_g'])
                }
            else:
                print("Please specify the correcct optimizer type")
                exit()
        else:
            print("Please specify the correcct method by model.set_method()")
            exit()


    def set_logger(self, exp=None):
        self.logger = exp

    def forward_step(self, h, x):
        return self.act(h @ self.W_hh + x @ self.W_xh + self.b_h)


    def backward_step(self, hp1, xp1):
        # given h_{t+1} and x_{t+1}, calculate h_{t}
        # self.W_xh is not trainable 
        return self.act(hp1 @ self.V_hh + xp1 @ self.W_xh.detach() + self.c_h)


    def validate(self, train_idx, val_loader):
        # select_idx = torch.randint(0, VAL_BATCH_NUM)
        with torch.no_grad():
            total_loss = 0
            total_samples = 0
            total_batches = 0
            total_correct = 0
            for i, (X, Y) in enumerate(val_loader):
                logit_out = self(X)
                # num_correct = utils.num_correct_samples(logit_out, Y)
                num_correct = utils.num_correct_samples(logit_out, Y)
                loss = self.cross_entropy_loss(logit_out.permute(1, 2, 0), Y)
                total_loss += loss
                total_samples += X.size(0)
                total_batches += 1
                total_correct += num_correct
            avg_loss = total_loss/total_batches
            accuracy = total_correct/total_samples
        if self.logger:
            self.logger.log_metric('val_loss', train_idx, avg_loss)
            self.logger.log_metric('val_acc', train_idx, accuracy)
        print(f'n_correct {total_correct}, total per batch {total_samples}')   
        print(f"Iteration {train_idx}                        -- val_loss:  {avg_loss.item():0.2f}  -- accuracy: {accuracy.item():0.4f}")
            # print("validation loss: ", avg_loss.item(), "  validation accuracy: ", accuracy.item())
        # return avg_loss, accuracy


    def train_step(self, batch_idx, X, Y):
        if self.method == "backprop":
            self.optimizer.zero_grad()
            logits = self(X.to(DEVICE)) # seq_len x batch_size x vocab_size
            loss = self.cross_entropy_loss(logits.permute(1, 2, 0), Y.to(DEVICE)) # Y: batch_size x vocab_size 
            loss.backward()
            self.optimizer.step()

            
            if self.logger:
                if  batch_idx == 0 or (batch_idx+1) % LOG_INVERVAL == 0:
                    self.logger.log_metric("training_loss", batch_idx, loss)
            if batch_idx == 0 or (batch_idx+1) % VAL_INTERVAL == 0:
                print(f"Iteration {batch_idx} -- train_loss:  {loss.item():0.2f}")
            # if (batch_idx+1) % VAL_INTERVAL == 0:
            #     val_loss, percent_correct = validate(model, val_loader)
            #     print(f"iteration {i+1}   train_loss {loss.item():.4f}    val_loss {val_loss:.4f}     \
            #             percent_correct {percent_correct:.4f}")
            #     neptune.log_metric("validation_loss", i, val_loss)
            #     neptune.log_metric("validation_accuracy", i, percent_correct)
            #     neptune.log_text("inputs", str(X[0,:]))
            #     # first_batch_pred = torch.transpose(logits[:,0,:], 0, 1)
            #     neptune.log_text("prediction", str(torch.argmax(logits[:,0,:], -1)))
            #     neptune.log_text("ground_truth", str(Y[0,:]))

        elif self.method == "targetprop":
            pass
        else:
            print("please set the correct method: model.set_method('backprop/targetprop')")
            exit()




# def train_backprop():
#     parser = argparse.ArgumentParser(description="Training a simple RNN model")
#     parser.add_argument('--seq_len', help='sequence length',
#                         type=int, required=True)
#     parser.add_argument('--lr', help='SGD learning rate',
#                         type=float, required=True)
#     # parser.add_argument('--momentum', help='SGD momentum',
#     #                     type=float, required=True)
#     args = parser.parse_args()
#     seq_len = args.seq_len
#     lr = args.lr
#     # momentum = args.momentum

#     train_loader, val_loader = datasets.prepare_copy_datasets(seq_len, TRAINING_BATCH_NUM * BATCH_SIZE, \
#         VAL_BATCH_NUM * BATCH_SIZE, BATCH_SIZE, DEVICE)

#     model = RNN(10, HIDDEN_SIZE, 10).to(DEVICE)
#     model.train()

#     criterion = torch.nn.CrossEntropyLoss()
#     # optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, nesterov=True)
#     optimizer = torch.optim.Adam(model.parameters(), lr=lr)

#     neptune.init(project_qualified_name="peterpdai/test")
#     neptune.create_experiment(
#         name=f'backprop_copymemory_len_{seq_len:02d}',
#         upload_source_files=['copymemory.py', 'utils.py', 'datasets.py'],
#         tags=["backprop", "training", 'copymemory'])

#     for i, (X, Y) in enumerate(train_loader):
#         optimizer.zero_grad()
#         logits = model(X.to(DEVICE)) # seq_len x batch_size x vocab_size
#         loss = criterion(logits.permute(1, 2, 0), Y.to(DEVICE)) # Y: batch_size x vocab_size 
#         loss.backward()
#         optimizer.step()

#         if (i+1) % LOG_INVERVAL == 0:
#             neptune.log_metric("training_loss", i, loss)

#         if (i+1) % VAL_INTERVAL == 0:
#             val_loss, percent_correct = validate(model, val_loader)
#             print(f"iteration {i+1}   train_loss {loss.item():.4f}    val_loss {val_loss:.4f}     \
#                     percent_correct {percent_correct:.4f}")
#             neptune.log_metric("validation_loss", i, val_loss)
#             neptune.log_metric("validation_accuracy", i, percent_correct)
#             neptune.log_text("inputs", str(X[0,:]))
#             # first_batch_pred = torch.transpose(logits[:,0,:], 0, 1)
#             neptune.log_text("prediction", str(torch.argmax(logits[:,0,:], -1)))
#             neptune.log_text("ground_truth", str(Y[0,:]))


# def tp_train(model, train_loader, val_loader, learning_rates, logger=None):

#     cross_entropy_loss = torch.nn.CrossEntropyLoss()
#     mse_loss = torch.nn.MSELoss()

#     opt_g = torch.optim.Adam([model.V_hh, model.c_h], lr=learning_rates['lr_g'])
#     opt_y = torch.optim.SGD([model.W_hy, model.b_y], lr=learning_rates['lr_y'])
#     opt_f = torch.optim.SGD([model.W_xh, model.W_hh, model.b_h], lr=learning_rates['lr_y'])

#     for iteration, (X, Y) in enumerate(train_loader):
#         X = one_hot(X.long().transpose(0, 1), num_classes=10).float()  # seq_len x batch_size x vocab_size
#         Y = Y.long() # batch_size x seq_len 
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
#                 f_loss_step = mse_loss(h_step, local_targets[step])
#                 f_loss_total += f_loss_step 
#                 # current_targets = prop_targets.pop()
#                 # assert (len(current_targets) == seq_len-step)
#                 # for i, target in enumerate(current_targets):
#                 #     # loss = (1/(seq_len-i))*mse_loss(h_step, target)
#                 #     loss = mse_loss(h_step, target)
#                 #     if (step==5) and (i == 0):
#                 #         logger.log_metric("step_loss", iteration, loss)
#                 #     f_loss_total += loss
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



def train_eval():
    parser = argparse.ArgumentParser(description="Training a simple RNN model")
    parser.add_argument('--method', help='backprop or targetprop', required=True)
    parser.add_argument('--task', help='expandsequence or copymemory', required=True)
    parser.add_argument('--seq_len', help='sequence length', type=int, required=True)

    ###########################
    ##### Hyperparameters #####
    ###########################

    # args from user
    args = parser.parse_args()
    method = args.method
    task = args.task 
    seq_len = args.seq_len 

    # generally fixed parameters
    if task == "expandsequence":
        in_dim = 5
        out_dim = 5
    elif task == "copymemory":
        in_dim = 10
        out_dim = 10
    else:
        print("Please input correct task name")
        exit()

    hidden_dim = 200
    actv_fn_type = 'relu'
    batch_size = 20
    optimizer_type = 'adam'

    # searchable parameters
    if method == 'backprop':
        learning_rate = 0.001 # for adam
    else: 
        learning_rate = {'lr_f': 0.001, 'lr_g': 0.001}

    hparams = {
        'method': method,
        'task': task,
        'seq_len': seq_len,
        'hidden_dim': hidden_dim,
        'actv_fn': actv_fn_type,
        'batch_size': batch_size,
        'optimizer': optimizer_type,
        'learning_rate': learning_rate
    }

    ###########################
    ##### Dataset ############
    ###########################

    # train_loader, val_loader = datasets.prepare_dataset(task, seq_len, batch_size)
    # train_loader, val_loader = datasets.prepare_copy_datasets(seq_len, TRAINING_BATCH_NUM * batch_size, VAL_BATCH_NUM * batch_size, batch_size, DEVICE)

    train_loader, val_loader = datasets.expand_seq_dataset(
        in_dim,  # including special blank character
        seq_len,
        2,
        batch_size,
        TRAINING_BATCH_NUM * batch_size,
        VAL_BATCH_NUM * batch_size ,
        DEVICE)
    ###########################
    ##### Model ###############
    ###########################

    model = RNN(in_dim, hidden_dim, out_dim, actv_fn_type)
    model.to(DEVICE)
    

    ###########################
    ##### Optimizer ###########
    ###########################

    model.set_method(method) # set attr for different training purpose: backprop or targetprop
    model.set_optimizer(optimizer_type, learning_rate) # accessed through model.optimizer

    ###########################
    ##### Logger ##############
    ###########################

    project = neptune.init(project_qualified_name="peterpdai/test")
    exp = project.create_experiment(
        name=f'run_{method}_{task}_{seq_len:02d}',
        upload_source_files=['copymemory.py', 'utils.py', 'datasets.py'],
        params = hparams,
        tags=[method, task, seq_len])

    model.set_logger(exp) 
    # model.set_logger() # if not using neptune, and just want to print it out in terminal 

    ###########################
    ##### Training loop #######
    ###########################

    model.train()
    for i, (X, Y) in enumerate(train_loader):
        model.train_step(i, X, Y)
        # print("trainign_step", i)
        if i == 0 or (i+1) % VAL_INTERVAL == 0:
            model.validate(i, val_loader)


if __name__ == "__main__":
    train_eval()
