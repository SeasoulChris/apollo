"""
Optuna example that optimizes a simple quadratic function.
In this example, we optimize a simple quadratic function. We also demonstrate how to continue an
optimization and to use timeouts.
We have the following two ways to execute this example:
(1) Execute this code directly.
    $ python quadratic_simple.py
(2) Execute through CLI.
    $ STUDY_NAME=`optuna create-study --storage sqlite:///example.db`
    $ optuna study optimize quadratic_simple.py objective --n-trials=100 --study $STUDY_NAME \
      --storage sqlite:///example.db
"""

import optuna


def objective(trial):
    x = trial.suggest_uniform('x', -10, 10)
    return (x - 2) ** 2


if __name__ == "__main__":
    # Let us minimize the objective function above.
    print("Running 10 trials...")
    study = optuna.create_study(study_name='distributed-example', storage='sqlite:///example.db')
    study.optimize(objective, n_trials=100)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    # We can continue the optimization as follows.
    print("Running 20 additional trials...")
    study.optimize(objective, n_trials=20)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))

    # We can specify the timeout instead of a number of trials.
    print("Running additional trials in 2 seconds...")
    study.optimize(objective, timeout=2.0)
    print("Best value: {} (params: {})\n".format(study.best_value, study.best_params))
