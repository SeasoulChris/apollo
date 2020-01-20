import optuna


def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2


study = optuna.create_study()


study.optimize(objective, n_trials=100)


study.best_params

study.best_value

study.best_trial
