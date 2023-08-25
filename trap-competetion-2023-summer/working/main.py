import json
import re
from os import path

import bert
import lightgbm as lgb
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn import decomposition, metrics, model_selection, preprocessing
from torch import nn
from tqdm import tqdm

tqdm.pandas()

SEED = 42

episodes_large = 0
members_small = 0
scaler = preprocessing.StandardScaler()

# BSV as DistributedDataParallel
bsv = bert.BertSequenceVectorizer()
# cuda.set_device(0)
# os.environ["MASTER_ADDR"] = "localhost"
# os.environ["MASTER_PORT"] = "12355"
# os.environ["NCCL_P2P_DISABLE"] = "1"
# os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
# rank = 0
# world_size = cuda.device_count()
# distributed.init_process_group("nccl", rank=rank, world_size=world_size)
# bsv_parallel = nn.parallel.DistributedDataParallel(bsv)
# print("DistributedDataParallel initialized.")
bsv_parallel = nn.DataParallel(bsv)

svd_n_components = 50
svd = decomposition.TruncatedSVD(n_components=svd_n_components, random_state=SEED)
bert_array = np.zeros((1, 768))


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

    # title, synopsisの単語数をカウント
    joined["title_word_count"] = (
        joined["title"].fillna("").apply(lambda x: len(x.split()))
    )
    joined["synopsis_word_count"] = (
        joined["synopsis"].fillna("").apply(lambda x: len(x.split()))
    )

    # # titleをbertでベクトル化
    # global bsv_parallel
    # global svd
    # global bert_array
    # title_features: pd.Series = (
    #     joined["title"]
    #     .fillna("")
    #     .progress_apply(
    #         lambda x: bsv_parallel.module.forward(bsv.vectorize(x))
    #     )  # TODO: 生のmoduleを使った方が速い、本来はDataParallelを使うべき
    # )
    # joined = joined.drop(columns=["title"])
    # bert_array = np.zeros((len(joined), 768))
    # for i, title_feature in enumerate(title_features):
    #     bert_array[i] = title_feature
    # title_vecs = pd.DataFrame(
    #     svd.fit_transform(bert_array),
    #     columns=[f"title_{i}" for i in range(svd_n_components)],
    # )
    # joined = pd.concat([joined, title_vecs], axis=1)

    # # synopsisをbertでベクトル化
    # synopsis_features: pd.Series = (
    #     joined["synopsis"]
    #     .fillna("")
    #     .progress_apply(lambda x: bsv_parallel.module.forward(bsv.vectorize(x)))
    # )
    # joined = joined.drop(columns=["synopsis"])
    # bert_array = np.zeros((len(joined), 768))
    # for i, synopsis_feature in enumerate(synopsis_features):
    #     bert_array[i] = synopsis_feature
    # synopsis_vecs = pd.DataFrame(
    #     svd.fit_transform(bert_array),
    #     columns=[f"synopsis_{i}" for i in range(svd_n_components)],
    # )
    # joined = pd.concat([joined, synopsis_vecs], axis=1)

    # TODO: BERTを復活させたら消す
    joined = joined.drop(columns=["title", "synopsis"])

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
    global episodes_large
    if is_train:
        episodes_large = joined["episodes"].quantile(0.99)
    joined["episodes"] = joined["episodes"].apply(
        lambda x: x if x < episodes_large else None
    )

    # 小さすぎるmembersをNoneにする
    global members_small
    if is_train:
        members_small = joined["members"].quantile(0.02)
    joined["members"] = joined["members"].apply(
        lambda x: x if x > members_small else None
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
        if rows.dtype == "int64" or rows.dtype == "float64":
            joined[col] = rows.fillna(rows.mean())

    # idの削除
    joined = joined.drop(columns=["user", "id", "anime_id"])

    # TODO: dtype == objectはとりあえずdropする
    for col in joined.columns:
        if joined[col].dtype == "object":
            joined = joined.drop(columns=[col])

    # 標準化
    x, y = joined, None
    if is_train:
        x, y = joined.drop(columns=["score"]), joined["score"]
        scaler.fit(x)
    x = pd.DataFrame(scaler.transform(x), columns=x.columns)

    x = pd.concat([x, genderOneHot, genreOneHot], axis=1)

    return x, y


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

    train_x, train_y = preprocess(csv_train, csv_anime, csv_profile, is_train=True)
    test_x, _ = preprocess(csv_test, csv_anime, csv_profile, is_train=False)

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
