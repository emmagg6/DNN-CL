from train_rnn import *

# proposed a set of hyperparameter using optuna.trial


def objective(trial, task, seq_len, method):

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

    # searchable parameters
    if method == 'backprop':
        opt_params = {'lr': trial.suggest_float("backprop_lr", 1e-5, 1, log=True),
                      'betas': (0.9, 0.999)}

    if method == 'targetprop':
        opt_params = {
            'opt_f': {
                'lr': trial.suggest_float("lr_f", 1e-5, 1, log=True),
                'betas': (0.9, 0.999)
            },
            'opt_g': {
                'lr': trial.suggest_float("lr_g", 1e-5, 1, log=True),
                'betas': (0.9, 0.999)
            },
            'lr_i': trial.suggest_float("lr_i", 1e-5, 1, log=True)
            # the initial step used to control how far the intial targets are from the activations
        }

    # All parameters required for training
    hparams_dict = {
        'task': task,
        'seq_len': seq_len,
        'method': method,
        'in_dim': in_dim,
        'out_dim': out_dim,
        'device': DEVICE,
        'hidden_dim': 128,
        'actv_fn': 'tanh',
        'batch_size': 20,
        'opt_params': opt_params,
    }

    return train_with_hparams(hparams_dict, logger=None, optuna_trial=trial)


def run_hyperparams_search():
    parser = argparse.ArgumentParser(description="Training a simple RNN model")
    parser.add_argument('--task', help='expandsequence or copymemory', required=True)
    parser.add_argument('--seq_len', help='sequence length', type=int, required=True)
    parser.add_argument('--method', help='backprop or targetprop', required=True)

    # --logger --> using neptune logger
    parser.add_argument('--logger', action='store_true', help='use neptune logger')

    # args from user
    args = parser.parse_args()
    if args.logger:
        neptune.init(project_qualified_name="peterpdai/tarseq-hparams-search")
        neptune.create_experiment(name=f'optuna-sweep_length_{args.seq_len:02d}',
                                  upload_source_files=['train_rnn_hparams_search.py', 'utils.py', 
                                                       'datasets.py', 'train_rnn.py'],
                                  tags=["rnn", "hyperparameter-search"])
        neptune_callback = opt_utils.NeptuneCallback(log_study=True, log_charts=True)
    else:
        neptune_callback = None

    study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler())
    study.optimize(lambda trial: objective(trial, args.task, args.seq_len,
                                           args.method), n_trials=50, callbacks=[neptune_callback])

    pruned_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.PRUNED]
    complete_trials = [t for t in study.trials if t.state == optuna.trial.TrialState.COMPLETE]

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
    run_hyperparams_search()
