
import random 
import numpy as np 
import torch
from torch import nn 
import pytorch_lightning as pl
import os 
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.profiler import AdvancedProfiler
from pytorch_lightning.loggers.neptune import NeptuneLogger
from pytorch_lightning import Trainer, seed_everything
from torch import optim
from torch.optim.optimizer import Optimizer
import hydra
# import neptune
import utils 
from omegaconf import DictConfig, OmegaConf
import datamodules as dm 

# seed_everything(42)

# sets seeds for numpy, torch, python.random and PYTHONHASHSEED.

############################################
######### Define model and system ###########
############################################

class LitRNN(pl.LightningModule):
    
    def __init__(self, method, in_dim, hidden_dim, out_dim, actv_fn, update_g_only_before, opt_params, optuna_trial=None):

        # store hyperparameters 
        super(LitRNN, self).__init__()
        self.method = method 
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim 
        self.actv_fn = actv_fn 
        self.update_g_only_before = update_g_only_before 
        self.opt_params = opt_params 
        self.optuna_trial = optuna_trial 

        self.automatic_optimization = False 

        # init hidden layer activation function 
        if self.actv_fn == 'tanh':
            self.act = nn.Tanh()    
        if self.actv_fn == 'relu':
            self.act = nn.ReLU()

        # loss functions 
        self.ce_loss = nn.CrossEntropyLoss()
        self.mse_loss = nn.MSELoss()

        # learnable parameters 
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

        if method == 'targetprop':
            self.V_hh = nn.Parameter(utils.rand_ortho(
                (self.hidden_dim, self.hidden_dim),
                np.sqrt(6. / (self.hidden_dim + self.hidden_dim))))
            self.c_h = nn.Parameter(torch.zeros([self.hidden_dim]))

    def f_func(self, h, x): # step forward function
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

    def forward(self, x):
        # x of size [batch_size, seq_len] of torch.long
        # conver to [seq_len, batch_size, vocab_size] of float
        x = nn.functional.one_hot(x.transpose(0, 1)).float()  # seq_len x batch_sz x vocab_size
        seq_len, batch_size, vocab_size = x.size()
        h = torch.zeros(batch_size, self.hidden_dim, device=self.device) 
        out_logits = []
        for step in range(seq_len):
            x_step = x[step, :, :]
            h = self.f_func(h, x_step) # post_activation 
            out_logits.append(self.y_func(h))
        return torch.stack(out_logits, dim=0).squeeze_().transpose(0, 1)  # batch_sz x seq_len x vocab_size


    # def backprop_method_step(self, batch, batch_idx):

    def training_step(self, batch, batch_idx, optimizer_idx):
        
        if self.method == 'backprop':
            # backward optimizer defined in configure_optimizers()
            opt = self.optimizers(use_pl_optimizer=True)
            assert (isinstance(opt, Optimizer))
            opt.zero_grad()

            X, Y = batch
            logits = self(X) # logits, batch_size * seq_len * vocab size
            loss = self.ce_loss(logits.permute(0, 2, 1), Y) # (minibatch, C, d1, d2, ...)
            self.manual_backward(loss)
            opt.step()

            self.log('train_loss', loss, on_step=True, prog_bar=True, logger=True)
            
        if self.method == 'targetprop':
            X, Y = batch 
            batch_X = nn.functional.one_hot(X.transpose(0, 1)).float()
            opt_f, opt_g = self.optimizers()
            # opt_f: update forward functions: W_hh, b_h, W_xh, W_hy, b_y
            # opt_g: update inverse function: V_hh, c_h
            batch_size, seq_len = Y.size()

            if batch_idx <= self.update_g_only_before: # update g_fun only before this number of steps 
                h_t = torch.zeros([batch_size, self.hidden_dim], dtype=torch.float, device=self.device)
                g_loss = torch.zeros(1, device=self.device)
                f_loss = torch.zeros(1, device=self.device)
                opt_g.zero_grad()
                h_prev = None 
                for t in range(seq_len):
                    x_t = batch_X[t, :, :]
                    if t == 0:
                        with torch.no_grad():
                            h_prev = self.f_func(h_t, x_t)
                    else: 
                        with torch.no_grad():
                            h_t = self.f_func(h_prev, x_t) 
                        rec = self.g_func(h_t, x_t) # trainable V_hh, c_h 
                        g_loss_t = self.mse_loss(h_prev, rec)
                        g_loss += g_loss_t 
                        h_prev = h_t 
                self.manual_backward(g_loss, opt_g)
                opt_g.step()
                opt_g.zero_grad()
                self.log('g_loss', g_loss, on_step=True, prog_bar=True, logger=True)

            else: # update both f and g at each training step 
                lr_i = self.opt_params['lr_i']
                h_t = torch.zeros([batch_size, self.hidden_dim],
                                  dtype=torch.float,
                                  device=self.device)
                Hs = [] # forward activations 
                local_targets = [] # local targets for training target prop 
                g_loss = torch.zeros(1, device=self.device)
                f_loss = torch.zeros(1, device=self.device)
                opt_g.zero_grad()
                opt_f.zero_grad()

                # forward pass to get 
                #  - forard activations Hs: List(Tensor)
                #  - local targets: local_targets: List(Tensor)
                #  - rec loss: g_loss 
                for t in range(seq_len):

                    # calculating forward activations 
                    x_t = batch_X[t, :, :]
                    h_t = self.f_func(h_t.detach(), x_t) # breaking the gradient backprop chain
                    Hs.append(h_t)

                    # calculating local target w.r.t. step-wise loss 
                    h_t_ = h_t.detach().clone().requires_grad_()
                    y_pred_t = self.y_func(h_t_) # y_pred:t: batch_size x vocab_size 
                    y_loss_t = self.ce_loss(y_pred_t, Y[:, t])
                    self.manual_backward(y_loss_t) 
                    # accumulate grad for 
                    #   - W_hy, b_y (opt_f.step())
                    #   - h_t_

                    with torch.no_grad():
                        target_t = h_t_ - lr_i * h_t_.grad 
                        local_targets.append(target_t)

                    # calculate rec loss for prev step 
                    if t > 0:
                        rec = self.g_func(h_t.detach(), x_t) # trainable: V_hh, c_h 
                        g_loss += self.mse_loss(rec, Hs[-2].detach()) 
                assert (torch.count_nonzero(self.W_hy.grad) > 0) # grad are accumlated for W_hy and b_y
                assert (torch.count_nonzero(self.b_y.grad) > 0)

                # Below is main step of the algorithm:
                # backward pass to accumulate the targets with different target propagation
                # activations are in Hs list, and local targets are in local_targets list, W_hy and b_y grad are
                # accmulated in the variables throughout the forward pass
                target_t = None
                for t_ in range(start=seq_len-1, stop=-1, step=-1): 
                    if t_ == seq_len - 1:
                        target_t = local_targets[t_] # last step, target is local_target
                        assert (target_t.requires_grad == False)
                        f_loss += self.mse_loss(Hs[t_], target_t)
                    else:
                        with torch.no_grad():
                            # targets backproped from the future
                            back_target = self.g_func(target_t, batch_X[t_+1, :, :]) # without linear correction
                            linear_correct = Hs[t_] - self.g_func(Hs[t_+1], batch_X[t_+1, :, :]) # linear correction
                            back_target += linear_correct

                            # combined future backproped targets and local targets
                            target_t = back_target + (local_targets[t_] - Hs[t_])/(t_ + 1)
                        assert (target_t.requries_grad == False)
                        f_loss += self.mse_loss(Hs[t_], target_t)

                # Update forward and backward functions 
                self.manual_backward(loss=g_loss, optimizer=opt_g, retain_graph=True)
                self.manual_backward(loss=f_loss, optimizer=opt_f) 

                opt_g.step()
                opt_f.step() # also using accumulated grad of W_hy, b_y

                opt_g.zero_grad()
                opt_f.zero_grad()
                        
                self.log('f_loss', f_loss, on_step=True, prog_bar=True, logger=True)  
                self.log('g_loss', g_loss, on_step=True, prog_bar=True, logger=True)
                

    def validation_step(self, batch, batch_idx):
        X, Y = batch 
        logit_out = self(X)
        n_correct = utils.num_correct_samples(logit_out, Y) # num correct predictiosn per batch 
        acc = n_correct/Y.size(0) # batch accuracy 
        loss = self.ce_loss(logit_out.permute(0, 2, 1), Y)
        self.log('val_loss', value=loss, logger=True, on_epoch=True)
        self.log('val_acc', value=acc, logger=True, on_epoch=True)
    
    def test_step(self, batch, batch_idx):
        pass 


    def configure_optimizers(self):

        if self.method == 'backprop':
            opt_bp = optim.Adam(self.parameters(), 
                              lr=self.opt_params.opt_bp.lr, 
                              betas=(self.opt_params.opt_bp.beta1, self.opt_params.opt_bp.beta2)
                              )
            return opt_bp
        if self.method == 'targetprop':
            opt_f = optim.Adam([self.W_xh, self.W_hh, self.b_h, self.W_hy, self.b_y],
                    lr=self.opt_params.opt_f.lr,
                    betas=(self.opt_params.opt_f.beta1,
                           self.opt_params.opt_f.beta2))
            opt_g = optim.Adam([self.V_hh, self.c_h],
                           lr=self.opt_params.opt_g.lr,
                           betas=(self.opt_params.opt_g.beta1,
                                  self.opt_params.opt_g.beta2))
            return [opt_f, opt_g] # They can be accessed through self.optimizers(use_pl_optimizer=True)


############################################
######### Training function ################
############################################


@hydra.main(config_path='conf', config_name="training_config")
def run_training(cfg: DictConfig):


    ######## Prepare data #######

    data_module=None 
    if cfg.dataset.name == 'copymemory':
        # data_module = dm.CopyMemoryDataModule(
        #     delay=cfg.dataset.seq_len,
        #     batch_size=cfg.batch_size,
        #     n_train=cfg.dataset.n_train,
        #     n_val=cfg.dataset.n_val,
        #     n_test=cfg.dataset.n_test,
        #     random_seeds={
        #         'train': random.randint(1, 10000),
        #         'val': random.randint(10001, 20000),
        #         'test': random.randint(20001, 30000) 
        #     }
        # )
        data_module = dm.CopyMemoryDataModule(
            delay=cfg.dataset.seq_len,
            rand = 1,
            T = 10
            # batch_size=cfg.batch_size,
            # n_train=cfg.dataset.n_train,
            # n_val=cfg.dataset.n_val,
            # n_test=cfg.dataset.n_test,
            # random_seeds={
            #     'train': random.randint(1, 10000),
            #     'val': random.randint(10001, 20000),
            #     'test': random.randint(20001, 30000) 
            # }
        )
    if cfg.dataset.name == 'expandsequence':
        data_module = dm.ExpandSequenceDataModule(
            seq_len=cfg.dataset.seq_len,
            vocab_size=cfg.dataset.input_dim,
            batch_size=cfg.batch_size,
            n_train=cfg.dataset.n_trai,
            n_val=cfg.dataset.n_va,
            n_test=cfg.dataset.n_test,
            random_seeds={
                'train': random.randint(1, 10000), 
                'val': random.randint(10001, 20000), 
                'test': random.randint(20001, 30000) 
                }
        )
    if cfg.dataset.name == 'nextchar':
        data_module = dm.NextCharDataModule(
            seq_len=cfg.dataset.seq_len,
            batch_size=32
        )




    ###### Define model and training system ####
    model = LitRNN(
        method=cfg.method.name,
        in_dim = cfg.dataset.input_dim,
        hidden_dim=cfg.hidden_dim,
        out_dim=cfg.dataset.output_dim,
        actv_fn=cfg.actv_fn,
        update_g_only_before=cfg.g_only_before_this,
        opt_params=cfg.method # contains learning information 
    )


    ####### init neptune logger ##########
    if cfg.use_neptune_logger:
        neptune.init(project_qualified_name=cfg.neptune_project)
        neptune_logger = neptune.create_experiment(
            name=cfg.neptune_exp.name,
            upload_source_files=cfg.neptune_exp.upload_files,
            params=dict(cfg)
        )
    else: 
        neptune_logger = None 


    ####### Create custom callbacks ######

    ####### Init pl trainer #########
    trainer = Trainer(
        track_grad_norm=2,
        weights_summary='top',
        progress_bar_refresh_rate=20,
        profiler=True,
        # check_val_every_n_epochs=500,
        # num_sanity_val_steps=2,
        gpus=1,
        auto_select_gpus=True,
        # fast_dev_run=True,
        overfit_batches=1,
        logger=neptune_logger,
        log_every_n_steps=100,
        # automatic_optimization=False
    )

    trainer.fit(
        model=model, 
        datamodule=data_module
        )



############################################
######### Run training #####################
############################################

if __name__ == '__main__':
    run_training()