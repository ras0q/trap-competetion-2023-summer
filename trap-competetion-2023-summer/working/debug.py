from os import path

import lightgbm as lgb
import matplotlib.pyplot as plt
import pandas as pd
import sklearn.model_selection as ms


def plot_validation_curve(
    model: lgb.LGBMRegressor,
    train_x: pd.DataFrame,
    train_y: pd.Series,
    temp_dir: str,
    cv: ms.KFold,
    random_state: int = 42,
):
    params = {
        "max_depth": [-1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
        "colsample_bytree": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        "reg_alpha": [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
        "reg_lambda": [0, 0.0001, 0.0003, 0.001, 0.003, 0.01, 0.03, 0.1, 0.3, 1, 3, 10],
        # "subsample": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        # "subsample_freq": [0, 1, 2, 3, 4, 5, 6, 7],
        "min_child_samples": [0, 2, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        "num_leaves": [2, 4, 8, 16, 32, 64, 96, 128, 160, 192, 224, 256],
    }

    for i, (k, v) in enumerate(params.items()):
        print(f"param: {k}, values: {v}")

        train_x, valid_x, train_y, valid_y = ms.train_test_split(
            train_x, train_y, test_size=0.2, random_state=random_state
        )

        train_scores, valid_scores = ms.validation_curve(
            model,
            train_x,
            train_y,
            param_name=k,
            param_range=v,
            cv=cv,
            scoring="neg_mean_squared_error",
            n_jobs=1,
            fit_params={
                "eval_set": [(valid_x, valid_y), (train_x, train_y)],
                "callbacks": [
                    lgb.early_stopping(50, first_metric_only=True, verbose=True)
                ],
            },
        )

        train_scores = -train_scores
        valid_scores = -valid_scores

        plt.plot(
            v, train_scores.mean(axis=1), label="train", color="orange", marker="o"
        )
        plt.fill_between(
            v,
            train_scores.mean(axis=1) + train_scores.std(axis=1),
            train_scores.mean(axis=1) - train_scores.std(axis=1),
            alpha=0.2,
            color="orange",
        )
        plt.plot(v, valid_scores.mean(axis=1), label="valid", color="blue", marker="o")
        plt.fill_between(
            v,
            valid_scores.mean(axis=1) + valid_scores.std(axis=1),
            valid_scores.mean(axis=1) - valid_scores.std(axis=1),
            alpha=0.2,
            color="blue",
        )
        plt.title(f"validation curve: {k}")
        plt.legend()
        plt.savefig(path.join(temp_dir, f"validation_curve_{k}.png"))
        plt.clf()
