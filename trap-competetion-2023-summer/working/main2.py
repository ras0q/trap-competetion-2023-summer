from os import path

import lightgbm as lgb
import numpy as np
import pandas as pd
import sklearn.metrics as mt
import sklearn.model_selection as ms
import sklearn.preprocessing as pp
from matplotlib import pyplot as plt

# global variables
SEED = 42


def preprocess(
    base: pd.DataFrame,
    anime: pd.DataFrame,
    profile: pd.DataFrame,
    is_train: bool,
):
    # 欠損値を表示
    print(f"is_train: {is_train}, isnull().sum:\n{base.isnull().sum()}")

    # userを整数でラベル化
    le = pp.LabelEncoder()
    base["user_label"] = le.fit_transform(base["user"])

    x_valid_columns = [
        "user_label",
    ]

    x = base[x_valid_columns]
    y = base["score"] if is_train else None

    return x, y


def train_predict(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    output_dir: str,
    n_split: int,
):
    model = lgb.LGBMRegressor(
        objective="regression",
        metric="mse",
        random_state=SEED,
        n_estimators=10000,
        verbose=-1,
        early_stopping_round=50,
    )

    val_preds = []
    test_preds = []
    result_file = open(path.join(output_dir, "result.txt"), "w")

    kf = ms.KFold(n_splits=n_split, shuffle=True, random_state=SEED)
    for i, (train_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
        _train_x = train_x.iloc[train_idx]
        _train_y = train_y.iloc[train_idx]
        _val_x = train_x.iloc[val_idx]
        _val_y = train_y.iloc[val_idx]

        model.fit(
            _train_x,
            _train_y,
            eval_set=[(_val_x, _val_y), (_train_x, _train_y)],
        )

        val_pred = model.predict(_val_x)
        val_preds.append(val_pred)

        val_loss = mt.mean_squared_error(_val_y, val_pred)
        msg = f"fold {i}: {val_loss}"
        print(msg)
        result_file.write(msg + "\n")

        test_pred = model.predict(test_x)
        test_preds.append(test_pred)

        lgb.plot_importance(model, importance_type="gain", max_num_features=20)
        plt.savefig(path.join(output_dir, f"feature_importance_{i + 1}.png"))

        lgb.plot_metric(model)
        plt.savefig(path.join(output_dir, f"metric_{i + 1}.png"))

    return val_preds, test_preds


def submit(sample_sub: pd.DataFrame, test_x: pd.DataFrame, preds: list):
    mean_pred = np.zeros(test_x.shape[0])

    for pred in preds:
        mean_pred += pred

    mean_pred /= len(preds)

    sample_sub["score"] = mean_pred
    return sample_sub


if __name__ == "__main__":
    input_dir = path.join(path.dirname(__file__), "..", "input")
    output_dir = path.join(path.dirname(__file__), "..", "output")

    csv_train = pd.read_csv(path.join(input_dir, "train.csv"))
    csv_test = pd.read_csv(path.join(input_dir, "test.csv"))
    csv_anime = pd.read_csv(path.join(input_dir, "anime.csv"))
    csv_profile = pd.read_csv(path.join(input_dir, "profile.csv"))
    csv_sample_sub = pd.read_csv(path.join(input_dir, "sample_submission.csv"))

    train_x, train_y = preprocess(csv_train, csv_anime, csv_profile, is_train=True)
    test_x, _ = preprocess(csv_test, csv_anime, csv_profile, is_train=False)

    val_preds, test_preds = train_predict(
        train_x, train_y, test_x, output_dir, n_split=5
    )

    sub = submit(csv_sample_sub, test_x, test_preds)
    plt.savefig(path.join(output_dir, "score.png"))
    sub.to_csv(path.join(output_dir, "submission.csv"), index=False)
