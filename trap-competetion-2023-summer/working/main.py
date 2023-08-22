import json
import re
from os import path

import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import metrics, model_selection, preprocessing

SEED = 42


def preprocess(
    base: pd.DataFrame,
    anime: pd.DataFrame,
    profile: pd.DataFrame,
    is_train: bool,
    scaler: preprocessing.StandardScaler = None,
):
    # id が重複している謎のデータの削除
    anime = anime.drop_duplicates(subset="id")

    # 列のマージ
    joined = base.merge(profile, on="user", how="left").merge(
        anime, left_on="anime_id", right_on="id", how="left"
    )
    joined = joined.drop(columns=["id", "anime_id"])

    # 誕生年だけを抽出
    def _get_birth_year(birthday):
        if type(birthday) != str:
            return None
        pattern = r"\b\d{4}\b"
        matches = re.findall(pattern, birthday)
        if len(matches) > 1:
            raise ValueError("find twice yaer")
        elif len(matches) == 0:
            return None
        else:
            return int(matches[0])

    joined["birth_year"] = joined["birthday"].apply(_get_birth_year)
    joined = joined.drop(columns=["birthday"])

    # start_day, end_dayを除外する
    joined = joined.drop(columns=["start_day", "end_day"])

    # 1-12以外のstart_month,end_monthをNoneにする
    joined["start_month"] = joined["start_month"].apply(
        lambda x: x if x in range(1, 13) else None
    )
    joined["end_month"] = joined["end_month"].apply(
        lambda x: x if x in range(1, 13) else None
    )

    # 大きすぎるepisodesをNoneにする
    _episodes_large = joined["episodes"].quantile(0.99)
    joined["episodes"] = joined["episodes"].apply(
        lambda x: x if x < _episodes_large else None
    )

    # 小さすぎるmembersをNoneにする
    _members_small = joined["members"].quantile(0.02)
    joined["members"] = joined["members"].apply(
        lambda x: x if x > _members_small else None
    )

    # genderの欠損値をOne-Hot Encodingで埋める
    genderOneHot = pd.get_dummies(joined["gender"].fillna("NaN"), dtype="uint8")
    joined = joined.drop(columns=["gender"])

    # genreをOne-Hot Encodingする
    joined["genre"] = joined["genre"].apply(
        lambda x: [] if type(x) != str else json.loads(x.replace("'", '"'))
    )
    mlb = preprocessing.MultiLabelBinarizer()
    _genres = mlb.fit_transform(joined["genre"])
    genreOneHot = pd.DataFrame(_genres, columns=mlb.classes_, dtype="uint8")
    joined = joined.drop(columns=["genre"])

    # NOTE: FOR DEBUG
    # for col in joined.columns:
    #     print(col)
    #     print(joined[col].describe())
    #     print("")

    # 残りの欠損値を平均で埋める
    for col in joined.columns:
        rows = joined[col]
        if type(rows[0]) == int or type(rows[0]) == float:
            rows = rows.fillna(rows.mean(), inplace=True)
        # TODO: type == objectの場合はとりあえずdropする
        elif type(rows[0]) == str:
            joined = joined.drop(columns=[col])

    # 標準化
    x, y = joined, None
    if is_train:
        x, y = joined.drop(columns=["score"]), joined["score"]
        if scaler is None:
            scaler = preprocessing.StandardScaler()
            scaler.fit(x)
        x = pd.DataFrame(scaler.transform(x), columns=x.columns)

    x = pd.concat([x, genderOneHot, genreOneHot], axis=1)

    return x, y, scaler


def train(
    train_x: pd.DataFrame,
    train_y: pd.Series,
    test_x: pd.DataFrame,
    output_dir: str,
    n_split: int,
):
    val_preds = np.zeros(len(train_x))
    preds = []

    result_file = open(path.join(output_dir, "result.txt"), "w")

    kf = model_selection.KFold(n_splits=n_split, shuffle=True, random_state=42)
    for i, (train_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
        model = lgb.LGBMRegressor(
            objective="regression",
            metric="mse",
            random_state=SEED,
            n_estimators=10000,
            verbose=-1,
            early_stopping_round=50,
        )

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
        val_preds[val_idx] = val_pred

        pred = model.predict(test_x)
        preds.append(pred)

        val_loss = metrics.mean_squared_error(_val_y, val_pred)
        print(f"Fold {i + 1} MSE: {val_loss:.4f}")
        result_file.write(f"Fold {i + 1} MSE: {val_loss:.4f}\n")

        lgb.plot_importance(
            model,
            importance_type="gain",
            title=f"Feature Importance - Fold {i + 1}",
            max_num_features=10,
        )
        plt.savefig(path.join(output_dir, f"feature_importance_{i + 1}.png"))

        lgb.plot_metric(model)
        plt.savefig(path.join(output_dir, f"metric_{i + 1}.png"))

    return val_preds, preds


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

    train_x, train_y, scaler = preprocess(
        csv_train, csv_anime, csv_profile, is_train=True
    )
    test_x, _, _ = preprocess(
        csv_test, csv_anime, csv_profile, is_train=False, scaler=scaler
    )

    # for col in train_x.columns:
    #     if train_x[col].dtype != "uint8":
    #         train_x[col].hist()
    #         plt.savefig(path.join(output_dir, f"tmp_{col}.png"))
    #         plt.clf()

    val_preds, preds = train(
        train_x,
        train_y,
        test_x,
        output_dir,
        n_split=4,
    )

    sub = submit(csv_sample_sub, test_x, preds)
    plt.savefig(path.join(output_dir, "score.png"))
    sub.to_csv(path.join(output_dir, "submission.csv"), index=False)
