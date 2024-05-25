# Runs of the main.py code


## USE

Settings for the arguments `--algorithm` and `--loss_feedback` used are shown below:
| Method            | `--algorithm` | `--loss_feedback` |
| ----------------- | ------------- | ----------------- |
| FW-DTP with BN    | FWDTP-BN      | any string        |
| DTP with BN       | DTP-BN        | DTP               |
| BP                | BP            | any string        |

Also more detailed setting are available by setting `--forward_function_1`,`--forward_function_2`,`--backward_function_1` and `--backward_function_2` (they are corresponding to f_mu, f_nu, g_mu and g_nu with the notations in Section 3.2).


Default architecture is set to the architecture used for MNIST in the paper. You can change the architecture by setting the arguments `--in_dim`, `--hid_dim`, `--out_dim` and `--depth` (they are corresponding to the input dimension, the number of hidden units, the output dimension and the number of layers).

| Dataset       | `--dataset`  |
| ------------- | ------------ |
| MNIST         | MNIST        |
| Fashion-MNIST | FashionMNIST |

When you change the dataset, you also should change the architecture since the input dimension `--in_dim` and the output dimension `--out_dim` are not automatically changed.

You can change the values of hyperparameters by setting the arguments as shown below:
| Parameter                             | Argument                   |
| ------------------------------------- | -------------------------- |
| Learning rate for feedforward network | `--learning_rate`          |
| Stepsize                              | `--stepsize`               |
| Learning rate for feedback network    | `--learning_rate_backward` |
| Standard deviation                    | `--std_backward`           |
| The feedback update frequency         | `--epochs_backward`        |


## Hyperparameter Optimisation from Paper:

LEARNING RATE HYPERPARAMETER FOR FW-DTP 
| Learning Rates                        | Best                      |
|-------------------------------------- | ------------------------- |
| {0.1, 0.2, 0.4, 0.8, 1, 2, 4, 8}      |                           |
| MNIST                                 | 0.1                       |
| F-MNIST                               | 1                         |

STEP SIZE HYPERPARAMETER FOR FW-DTP 
| Step Sizes                                            | Best                      |
|------------------------------------------------------ | ------------------------- |
| {0.001, 0.002, 0.004, 0.008, 0.01, 0.02, 0.04, 0.08}  |                           |
| MNIST                                                 | 0.04                      |
| F-MNIST                                               | 0.004                     |




## Continual Learning Runs

BASE : 

1. python main.py --algorithm=DTP-BN --dataset=MNIST --forward_last_activation=linear-BN --learning_rate=0.1 --stepsize=0.04 --test --log --save=yes --epochs=10
> checkpoint save name : DTP-mnist
> wandb : DTP-m

2. python main.py --algorithm=FWDTP-BN --dataset=MNIST --forward_last_activation=linear-BN --learning_rate=0.1 --stepsize=0.04 --test --log --save=yes --epochs=10
> checkpoint save name : FWDTP-mnist
> wandb : FWDTP-m

3. python main.py --algorithm=DTP-BN --dataset=FashionMNIST --forward_last_activation=linear-BN --learning_rate=1 --stepsize=0.004 --test --log --save=yes --epochs=10
> checkpoint save name : DTP-fmnist
> wandb : DTP-f

4. python main.py --algorithm=FWDTP-BN --dataset=FashionMNIST --forward_last_activation=linear-BN --learning_rate=1 --stepsize=0.004 --test --log --save=yes --epochs=10
> checkpoint save name : FWDTP-fmnist
> wandb : FWDTP-f

5. python main.py --algorithm=BP --dataset=MNIST --learning_rate=0.1 --stepsize=0.04 --test --log --save=yes --epochs=10
> forward : tanh-BN, orthogonal
> checkpoint save name : BP-mnist
> wandb : BP-m

6. python singleRun.py --algorithm=BP --dataset=FashionMNIST --learning_rate=1 --stepsize=0.004 --test --log --save=yes --epochs=10
> forward : tanh-BN, orthogonal
> checkpoint save name : BP-f
> wandb :
python singleRun.py --algorithm=BP --dataset=FashionMNIST --learning_rate=1 --stepsize=0.004 --test --save=no --epochs=10
python singleRun.py --algorithm=FWDTP-BN --dataset=FashionMNIST --learning_rate=1 --stepsize=0.004 --test --save=no --epochs=10

CONTINUAL LEARNING 1 :

1. python main.py --algorithm=DTP-BN --dataset=FashionMNIST --forward_last_activation=linear-BN --learning_rate=1 --stepsize=0.004 --test --log --save=yes --epochs=10 --continual=yes
> previous task : MNIST
> initialized save name : DTP-mnist
> checkpoint save name : DTP-m-f

2. python main.py --algorithm=FWDTP-BN --dataset=FashionMNIST --forward_last_activation=linear-BN --learning_rate=1 --stepsize=0.004 --test --log --save=yes --epochs=10 --continual=yes
> previous task : MNIST
> initialized save name : FWDTP-mnist
> checkpoint save name : FWDTP-m-f

3. python main.py --algorithm=BP --dataset=FashionMNIST --learning_rate=1 --stepsize=0.004 --test --log --save=yes --epochs=10 --continual=yes
> previous task : MNIST
> initialized save name : BP-m
> checkpoint save name : BP-m-f


CONTINUAL LEARNING 2 :
1. python main.py --algorithm=DTP-BN --dataset=MNIST --forward_last_activation=linear-BN --learning_rate=0.1 --stepsize=0.04 --test --log --save=yes --epochs=10 --continual=yes
> previous tasks : MNIST, FashionMNIST
> initialized save name : DTP-m-f
> checkpoint save name : DTP-m-f-m

2. python main.py --algorithm=FWDTP-BN --dataset=MNIST --forward_last_activation=linear-BN --learning_rate=0.1 --stepsize=0.04 --test --log --save=yes --epochs=10 --continual=yes
> previous tasks : MNIST, FashionMNIST
> initialized save name : FWDTP-m-f
> checkpoint save name : FWDTP-m-f-m

3. python main.py --algorithm=BP --dataset=MNIST --learning_rate=0.1 --stepsize=0.04 --test --log --save=yes --epochs=10 --continual=yes
> previous tasks : MNIST, FashionMNIST
> initialized save name : BP-m-f
> checkpoint save name : BP-m-f-m


