import optuna


def objective(trial):
    config = trial.suggest_uniform('x', -10, 10)

    tuner = MracAutoTuner()
    tuner.set_config(list_of_pairs)  # {path, config}
    tuner.main()
    # 1. Config generations
    # 2. Send / Call replay_engine
    # 3. Gather bags.
    # 4. Call profiling.
    # 5. Return weighted scores.
    list_of_scores = tuner.get_scores()  # {config, scenario -> score}
    return weighted_score(scores)
    # return (x - 2) ** 2


def weighted_score(scores):
    return 1


sampler_marc = MRACSampler()

study = optuna.create_study(sampler=sampler_marc)

# Optionally can run in parellel for trainig, using SQLite or MySQL.
study.optimize(objective, n_trials=100)


study.best_params

study.best_value

study.best_trial
