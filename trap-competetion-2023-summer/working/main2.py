import threading
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


def preprocess_anime(anime: pd.DataFrame):
    anime = anime.copy()

    # id が重複している謎のデータの削除
    anime = anime.drop_duplicates(subset="id")

    # rankedの欠損値を平均で補完
    _ranked_mean = anime["ranked"].mean()
    anime["ranked"] = anime["ranked"].fillna(_ranked_mean)

    # episodesの欠損値を中央値で補完
    _episode_median = anime["episodes"].median()
    anime["episodes"] = anime["episodes"].fillna(_episode_median)

    # yearとmonthを結合してstart_ym,end_ymを生成
    # ym = (year - 1900) * 12 + month
    # yearが欠損しているときはmonthに入っていることがある (monthも欠損しているときは平均で補完)
    # monthの欠損値は6で補完
    for p in ["start", "end"]:
        _year_mean = round(anime[f"{p}_year"].mean())
        for i, row in anime.iterrows():
            _y, _m = row[f"{p}_year"], row[f"{p}_month"]
            if pd.isna(_y):
                anime.loc[i, f"{p}_ym"] = (
                    _m if pd.isna(_m) and _m >= 1900 else _year_mean
                ) - 1900
            else:
                anime.loc[i, f"{p}_ym"] = (_y - 1900) * 12 + (
                    _m if pd.isna(_m) and 1 <= _m <= 12 else 6
                )

    # genreのダミー化
    genres = (
        pd.get_dummies(
            anime["genre"]
            .apply(eval)
            .apply(lambda x: ["NaN"] if len(x) == 0 else x)
            .apply(pd.Series)
            .stack(),
            prefix="genre",
            dtype="uint8",
        )
        .groupby(level=0)
        .sum()
    )
    anime = pd.concat(
        [anime, genres],
        axis=1,
    )

    # 標準化
    standardized_columns = [
        "ranked",
        "popularity",
        "members",
        "episodes",
        "start_ym",
        "end_ym",
    ]
    anime[standardized_columns] = scaler.fit_transform(anime[standardized_columns])

    return anime


def preprocess_profile(profile: pd.DataFrame):
    profile = profile.copy()

    # userを整数でラベル化
    le = pp.LabelEncoder()
    profile["user_label"] = le.fit_transform(profile["user"])

    # birthdayから誕生年(birth_year)を生成
    profile["birth_year"] = profile["birthday"].apply(myutil.get_birth_year)
    # 1935年以前のデータは適当に設定したと思われるので削除
    profile["birth_year"] = profile["birth_year"].apply(
        lambda x: x if x >= 1935 else None
    )
    # 欠損値を平均で補完
    _birth_year_mean = profile["birth_year"].mean()
    profile["birth_year"] = profile["birth_year"].fillna(_birth_year_mean)

    # genderのダミー化
    genders = pd.get_dummies(
        profile["gender"].fillna("NaN"), prefix="gender_", dtype="uint8"
    )
    profile = pd.concat(
        [profile, genders],
        axis=1,
    )

    # 標準化
    standardized_columns = [
        "birth_year",
    ]
    profile[standardized_columns] = scaler.fit_transform(profile[standardized_columns])

    return profile


def preprocess(
    base: pd.DataFrame,
    anime: pd.DataFrame,
    profile: pd.DataFrame,
    is_train: bool,
):
    # 列のマージ
    joined = base.merge(profile, on="user", how="left").merge(
        anime, left_on="anime_id", right_on="id", how="left"
    )

    # 範囲外のscoreを含む行を削除
    if is_train:
        joined = joined[(joined["score"] >= 1) & (joined["score"] <= 10)]

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
        "start_ym",
        "end_ym",
    ]

    # 分布,相関係数の可視化 (別スレッドで実行)
    def pairplot():
        cp = san.CustomPairPlot()
        cp.pairanalyzer(joined[x_valid_columns + ["score"]], diag_kind="hist")
        plt.savefig(path.join(output_dir, "pairplot.png"))
        plt.clf()

    if is_train:
        thread = threading.Thread(target=pairplot)
        thread.start()

    # 追加のダミーカラム
    x_valid_columns += [
        col
        for col in joined.columns
        if col.startswith("genre_") or col.startswith("gender_")
    ]

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
        max_depth=5,
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

    print("anime preprocessing...")
    anime = preprocess_anime(csv_anime)
    print("profile preprocessing...")
    profile = preprocess_profile(csv_profile)
    print("train preprocessing...")
    train_x, train_y = preprocess(csv_train, anime, profile, is_train=True)
    print("test preprocessing...")
    test_x, _ = preprocess(csv_test, anime, profile, is_train=False)

    val_preds, test_preds = train_predict(
        train_x, train_y, test_x, output_dir, n_split=5
    )

    sub = submit(csv_sample_sub, test_x, test_preds)
    plt.savefig(path.join(output_dir, "score.png"))
    sub.to_csv(path.join(output_dir, "submission.csv"), index=False)
