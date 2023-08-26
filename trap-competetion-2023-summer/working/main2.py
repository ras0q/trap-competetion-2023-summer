from os import path

import lightgbm as lgb
import matplotlib.pyplot as plt
import myutil
import numpy as np
import pandas as pd
import seaborn_analyzer as san
import sklearn.metrics as mt
import sklearn.model_selection as ms
import sklearn.preprocessing as pp

# global variables
SEED = 42
scaler = pp.StandardScaler()


def preprocess(
    base: pd.DataFrame,
    anime: pd.DataFrame,
    profile: pd.DataFrame,
    is_train: bool,
):
    # id が重複している謎のデータの削除
    anime = anime.drop_duplicates(subset="id")

    # 列のマージ
    joined = base.merge(profile, on="user", how="left").merge(
        anime, left_on="anime_id", right_on="id", how="left"
    )

    # 欠損値を表示
    print(f"is_train: {is_train}, isnull().sum:\n{joined.isnull().sum()}")

    # 範囲外のscoreを含む行を削除
    if is_train:
        joined = joined[(joined["score"] >= 1) & (joined["score"] <= 10)]

    # userを整数でラベル化
    le = pp.LabelEncoder()
    joined["user_label"] = le.fit_transform(joined["user"])

    # birthdayから誕生年(birth_year)を生成
    joined["birth_year"] = joined["birthday"].apply(myutil.get_birth_year)
    # 1935年以前のデータは適当に設定したと思われるので削除
    joined["birth_year"] = joined["birth_year"].apply(
        lambda x: x if x >= 1935 else None
    )
    # 欠損値を平均で補完
    _birth_year_mean = joined["birth_year"].mean()
    joined["birth_year"] = joined["birth_year"].fillna(_birth_year_mean)

    # rankedの欠損値を平均で補完
    _ranked_mean = joined["ranked"].mean()
    joined["ranked"] = joined["ranked"].fillna(_ranked_mean)

    # episodesの欠損値を中央値で補完
    _episode_median = joined["episodes"].median()
    joined["episodes"] = joined["episodes"].fillna(_episode_median)

    # yearが欠損しているときはmonthに入っていることがあるので補完する
    # monthも欠損しているときは平均で補完
    for p in ["start", "end"]:
        _year_mean = joined[f"{p}_year"].mean()
        for i, row in joined[joined[f"{p}_year"].isnull()].iterrows():
            if row[f"{p}_month"] is not None and row[f"{p}_month"] >= 1900:
                joined.loc[i, f"{p}_year"] = row[f"{p}_month"]
            else:
                joined.loc[i, f"{p}_year"] = _year_mean

    # genreのダミー化
    genres = (
        pd.get_dummies(
            joined["genre"].apply(eval).apply(pd.Series).stack(),
            prefix="genre_",
            dtype="uint8",
        )
        .groupby(level=0)
        .sum()
    )
    joined = pd.concat(
        [joined, genres],
        axis=1,
    )

    # genderのダミー化
    genders = pd.get_dummies(
        joined["gender"].fillna("NaN"), prefix="gender_", dtype="uint8"
    )
    joined = pd.concat(
        [joined, genders],
        axis=1,
    )

    # 標準化
    standardized_columns = [
        "ranked",
        "popularity",
        "birth_year",
        "members",
        "episodes",
        "start_year",
        "end_year",
    ]
    if is_train:
        scaler.fit(joined[standardized_columns])
    joined[standardized_columns] = scaler.transform(joined[standardized_columns])

    # 使用するカラムのリスト
    # 分布を表示しないものは後に追加する
    x_valid_columns = [
        "user_label",
        "anime_id",
        "ranked",
        "popularity",
        "birth_year",
        "members",
        "episodes",
        "start_year",
        "end_year",
    ]

    # 分布,相関係数の可視化
    if is_train:
        cp = san.CustomPairPlot()
        cp.pairanalyzer(joined[x_valid_columns + ["score"]], diag_kind="hist")
        plt.savefig(path.join(output_dir, "pairplot.png"))
        plt.clf()

    # 追加のダミーカラム
    x_valid_columns += genres.columns.tolist() + genders.columns.tolist()

    x = joined[x_valid_columns]
    y = joined["score"] if is_train else None

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
        num_leaves=8,
        min_child_samples=5,
        colsample_bytree=0.8,
        reg_alpha=0.3,
        reg_lambda=0.3,
        verbose=-1,
    )

    val_preds = []
    test_preds = []
    result_file = open(path.join(output_dir, "result.txt"), "w")

    kf = ms.KFold(n_splits=n_split, shuffle=True, random_state=SEED)

    # import debug
    # debug.plot_validation_curve(
    #     model, train_x, train_y, output_dir, cv=kf, random_state=SEED
    # )

    for i, (train_idx, val_idx) in enumerate(kf.split(train_x, train_y)):
        _train_x = train_x.iloc[train_idx]
        _train_y = train_y.iloc[train_idx]
        _val_x = train_x.iloc[val_idx]
        _val_y = train_y.iloc[val_idx]

        model.fit(
            _train_x,
            _train_y,
            eval_set=[(_val_x, _val_y), (_train_x, _train_y)],
            callbacks=[lgb.early_stopping(50, first_metric_only=True, verbose=True)],
        )

        val_pred = model.predict(_val_x)
        val_preds.append(val_pred)

        val_loss = mt.mean_squared_error(_val_y, val_pred)
        msg = f"fold {i}: {val_loss:.5f}"
        print(msg)
        result_file.write(msg + "\n")

        test_pred = model.predict(test_x)
        test_preds.append(test_pred)

        lgb.plot_importance(model, importance_type="gain", max_num_features=15)
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
