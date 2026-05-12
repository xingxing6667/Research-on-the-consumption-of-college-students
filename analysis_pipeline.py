from __future__ import annotations

import math
import os
import warnings
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "4")

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from torch import nn
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    mean_absolute_error,
    mean_squared_error,
    silhouette_score,
)
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.statespace.sarimax import SARIMAX

torch.set_num_threads(1)

ROOT = Path(__file__).resolve().parent
CONSUMPTION_XLSX = ROOT / "2023年9-12月理学院(1).xlsx"
EXTRACTED = ROOT / "output" / "extracted"
QUESTIONNAIRE_SUMMARY_CSV = EXTRACTED / "questionnaire_summary.csv"
OUT = ROOT / "output" / "analysis"
FIG = OUT / "figures"
ELBOW = OUT / "elbow_method"


def setup_plot_style() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    sns.set_theme(style="whitegrid", font="Microsoft YaHei")


def mode_or_first(values: pd.Series) -> str:
    clean = values.dropna()
    if clean.empty:
        return ""
    mode = clean.mode()
    if not mode.empty:
        return str(mode.iloc[0])
    return str(clean.iloc[0])


def read_extracted_questionnaire() -> pd.DataFrame:
    if not QUESTIONNAIRE_SUMMARY_CSV.exists():
        raise FileNotFoundError(
            f"Missing extracted questionnaire file: {QUESTIONNAIRE_SUMMARY_CSV}. "
            "Run `python extract_docx_data.py` before `python analysis_pipeline.py`."
        )
    return pd.read_csv(QUESTIONNAIRE_SUMMARY_CSV)


def merchant_category(name: str) -> str:
    text = str(name)
    if "水果" in text:
        return "水果"
    if any(key in text for key in ["米饭", "主食"]):
        return "主食米饭"
    if any(key in text for key in ["副食", "小炒", "凉菜", "米粉", "刀削面", "民族", "土耳其", "清真", "餐厅", "小厨"]):
        return "菜品餐食"
    return "其他"


def read_consumption() -> tuple[pd.DataFrame, dict[str, object]]:
    df = pd.read_excel(CONSUMPTION_XLSX, sheet_name="Sheet1")
    raw_rows = len(df)
    raw_students = df["学工号"].nunique()
    exact_duplicate_rows = int(df.duplicated().sum())
    potential_duplicate_without_index = int(df.drop(columns=["Unnamed: 0"], errors="ignore").duplicated().sum())
    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df["date"] = pd.to_datetime(df["日期"].astype(str), format="%Y%m%d", errors="coerce")
    df["amount"] = pd.to_numeric(df["交易金额"], errors="coerce")
    df = df.dropna(subset=["date", "amount"])
    df = df[df["amount"] > 0].copy()

    unique_ids = sorted(df["学工号"].astype(str).unique())
    id_map = {sid: f"S{idx:04d}" for idx, sid in enumerate(unique_ids, start=1)}
    df["student_id"] = df["学工号"].astype(str).map(id_map)

    df["weekday"] = df["date"].dt.weekday
    df["is_weekend"] = df["weekday"].isin([5, 6]).astype(int)
    df["day"] = df["date"].dt.day
    df["is_month_start_period"] = (df["day"] <= 10).astype(int)
    df["is_month_end_period"] = (df["day"] >= 21).astype(int)
    df["year_month"] = df["date"].dt.to_period("M").astype(str)
    df["merchant_category"] = df["商户"].map(merchant_category)

    meta = {
        "raw_rows": raw_rows,
        "clean_rows": len(df),
        "exact_duplicate_rows": exact_duplicate_rows,
        "potential_duplicate_without_index": potential_duplicate_without_index,
        "raw_students": raw_students,
        "students": df["student_id"].nunique(),
        "date_min": df["date"].min().date().isoformat(),
        "date_max": df["date"].max().date().isoformat(),
        "total_amount": float(df["amount"].sum()),
        "avg_transaction_amount": float(df["amount"].mean()),
    }
    return df, meta


def amount_share(df: pd.DataFrame, column: str, mapping: dict[str, str], prefix: str) -> pd.DataFrame:
    pivot = df.pivot_table(index="student_id", columns=column, values="amount", aggfunc="sum", fill_value=0)
    out = pd.DataFrame(index=pivot.index)
    total = pivot.sum(axis=1).replace(0, np.nan)
    for raw, clean in mapping.items():
        if raw in pivot.columns:
            out[f"{prefix}_{clean}_share"] = pivot[raw] / total
        else:
            out[f"{prefix}_{clean}_share"] = 0.0
    return out.fillna(0)


def mask_amount_share(df: pd.DataFrame, mask_col: str, out_col: str) -> pd.DataFrame:
    total = df.groupby("student_id")["amount"].sum()
    part = df[df[mask_col] == 1].groupby("student_id")["amount"].sum()
    share = (part / total).fillna(0)
    return share.rename(out_col).to_frame()


def normalized_entropy(values: pd.Series) -> float:
    total = values.sum()
    if total <= 0 or len(values) <= 1:
        return 0.0
    p = values / total
    return float(-(p * np.log(p)).sum() / np.log(len(values)))


def build_student_features(df: pd.DataFrame) -> pd.DataFrame:
    global_days = int((df["date"].max() - df["date"].min()).days) + 1
    base = df.groupby("student_id").agg(
        gender=("性别", mode_or_first),
        department=("部门", mode_or_first),
        total_amount=("amount", "sum"),
        txn_count=("amount", "size"),
        avg_amount=("amount", "mean"),
        median_amount=("amount", "median"),
        amount_std=("amount", "std"),
        active_days=("date", "nunique"),
        merchant_count=("商户", "nunique"),
        first_date=("date", "min"),
        last_date=("date", "max"),
    )
    base["amount_std"] = base["amount_std"].fillna(0)
    base["amount_cv"] = base["amount_std"] / base["avg_amount"].replace(0, np.nan)
    base["amount_cv"] = base["amount_cv"].replace([np.inf, -np.inf], np.nan).fillna(0)
    base["avg_daily_amount_window"] = base["total_amount"] / global_days
    base["avg_daily_txn_window"] = base["txn_count"] / global_days
    base["avg_active_day_amount"] = base["total_amount"] / base["active_days"].replace(0, np.nan)
    base["txn_per_active_day"] = base["txn_count"] / base["active_days"].replace(0, np.nan)

    meal_share = amount_share(
        df,
        "餐次",
        {"早餐": "breakfast", "午餐": "lunch", "晚餐": "dinner", "宵夜": "late_night"},
        "meal_amount",
    )
    pay_share = amount_share(
        df,
        "交易类型",
        {"扫码支付": "scan_pay", "电子账户消费": "e_account", "持卡人消费": "card"},
        "pay_amount",
    )
    category_share = amount_share(
        df,
        "merchant_category",
        {"主食米饭": "staple", "菜品餐食": "dish", "水果": "fruit", "其他": "other"},
        "merchant_amount",
    )

    weekend_share = mask_amount_share(df, "is_weekend", "weekend_amount_share")
    month_start_share = mask_amount_share(df, "is_month_start_period", "month_start_amount_share")
    month_end_share = mask_amount_share(df, "is_month_end_period", "month_end_amount_share")

    merchant_amounts = df.groupby(["student_id", "商户"])["amount"].sum()
    entropy = merchant_amounts.groupby(level=0).apply(normalized_entropy).rename("merchant_entropy_norm")

    monthly = df.pivot_table(index="student_id", columns="year_month", values="amount", aggfunc="sum", fill_value=0)
    monthly["monthly_amount_mean"] = monthly.mean(axis=1)
    monthly["monthly_amount_std"] = monthly.std(axis=1)
    monthly_features = monthly[["monthly_amount_mean", "monthly_amount_std"]]

    features = base.join(
        [
            meal_share,
            pay_share,
            category_share,
            weekend_share,
            month_start_share,
            month_end_share,
            entropy,
            monthly_features,
        ],
        how="left",
    )
    features = features.fillna(0)
    features["log_total_amount"] = np.log1p(features["total_amount"])
    features["log_txn_count"] = np.log1p(features["txn_count"])
    features["log_merchant_count"] = np.log1p(features["merchant_count"])
    return features.reset_index()


def add_elbow_metrics(eval_df: pd.DataFrame) -> pd.DataFrame:
    out = eval_df.sort_values("k").reset_index(drop=True).copy()
    out["inertia_drop"] = out["inertia"].shift(1) - out["inertia"]
    out["inertia_drop_rate"] = out["inertia_drop"] / out["inertia"].shift(1)
    out["elbow_score"] = out["inertia_drop"] - out["inertia_drop"].shift(-1)
    return out


def select_elbow_k(eval_df: pd.DataFrame) -> int:
    metrics = add_elbow_metrics(eval_df)
    candidates = metrics[(metrics["k"] >= 3) & metrics["elbow_score"].notna()].copy()
    if not candidates.empty and candidates["elbow_score"].max() > 0:
        return int(candidates.sort_values("elbow_score", ascending=False).iloc[0]["k"])

    # Fallback: use maximum distance to the line between first and last inertia points.
    x = metrics["k"].to_numpy(dtype=float)
    y = metrics["inertia"].to_numpy(dtype=float)
    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())
    p1 = np.array([x_norm[0], y_norm[0]])
    p2 = np.array([x_norm[-1], y_norm[-1]])
    line = p2 - p1
    distances = []
    for xi, yi in zip(x_norm, y_norm):
        p = np.array([xi, yi])
        distances.append(abs(np.cross(line, p - p1)) / np.linalg.norm(line))
    return int(x[int(np.argmax(distances))])


def name_clusters(profile: pd.DataFrame) -> dict[int, str]:
    names: dict[int, str] = {}
    order = profile.sort_values("total_amount", ascending=False).index.tolist()
    if not order:
        return names

    if len(order) == 2:
        levels = ["高消费", "低消费"]
    elif len(order) == 3:
        levels = ["高消费", "中消费", "低消费"]
    elif len(order) == 4:
        levels = ["高消费", "较高消费", "较低消费", "低消费"]
    elif len(order) == 5:
        levels = ["高消费", "较高消费", "中消费", "较低消费", "低消费"]
    else:
        levels = [f"第{rank}档消费" for rank in range(1, len(order) + 1)]

    for rank, idx in enumerate(order, start=1):
        names[idx] = f"类{rank}-{levels[rank - 1]}"
    return names


def cluster_students(features: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    cluster_cols = [
        "log_total_amount",
        "avg_amount",
        "amount_cv",
        "active_days",
        "txn_per_active_day",
        "log_merchant_count",
        "merchant_entropy_norm",
        "meal_amount_breakfast_share",
        "meal_amount_lunch_share",
        "meal_amount_dinner_share",
        "meal_amount_late_night_share",
        "pay_amount_scan_pay_share",
        "weekend_amount_share",
        "month_start_amount_share",
        "month_end_amount_share",
        "merchant_amount_staple_share",
        "merchant_amount_dish_share",
        "merchant_amount_fruit_share",
    ]

    X = features[cluster_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    eval_rows = []
    max_k = min(8, len(features) - 1)
    for k in range(2, max_k + 1):
        model = KMeans(n_clusters=k, init="k-means++", n_init=30, random_state=42)
        labels = model.fit_predict(X_scaled)
        eval_rows.append(
            {
                "k": k,
                "inertia": model.inertia_,
                "silhouette": silhouette_score(X_scaled, labels),
                "calinski_harabasz": calinski_harabasz_score(X_scaled, labels),
                "davies_bouldin": davies_bouldin_score(X_scaled, labels),
            }
        )
    eval_df = add_elbow_metrics(pd.DataFrame(eval_rows))
    best_k = select_elbow_k(eval_df)

    best_model = KMeans(n_clusters=best_k, init="k-means++", n_init=50, random_state=42)
    features = features.copy()
    features["cluster"] = best_model.fit_predict(X_scaled)

    profile_cols = [
        "total_amount",
        "txn_count",
        "avg_amount",
        "amount_cv",
        "active_days",
        "txn_per_active_day",
        "merchant_count",
        "merchant_entropy_norm",
        "meal_amount_breakfast_share",
        "meal_amount_lunch_share",
        "meal_amount_dinner_share",
        "meal_amount_late_night_share",
        "pay_amount_scan_pay_share",
        "weekend_amount_share",
        "merchant_amount_staple_share",
        "merchant_amount_dish_share",
        "merchant_amount_fruit_share",
    ]
    profile = features.groupby("cluster")[profile_cols].mean()
    size = features.groupby("cluster").size().rename("students")
    profile = profile.join(size)
    profile["student_share"] = profile["students"] / len(features)
    names = name_clusters(profile)
    profile["cluster_name"] = profile.index.map(names)
    features["cluster_name"] = features["cluster"].map(names)
    profile = profile.reset_index()

    return features, eval_df, profile, X_scaled


def build_daily_series(df: pd.DataFrame) -> pd.DataFrame:
    daily = df.groupby("date").agg(
        total_amount=("amount", "sum"),
        transaction_count=("amount", "size"),
        active_students=("student_id", "nunique"),
    )
    idx = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    daily = daily.reindex(idx, fill_value=0)
    daily.index.name = "date"
    daily["weekday"] = daily.index.weekday
    daily["is_weekend"] = daily["weekday"].isin([5, 6]).astype(int)
    daily["month"] = daily.index.month
    daily["day"] = daily.index.day
    return daily.reset_index()


def forecast_metrics(y_true: pd.Series, y_pred: pd.Series) -> dict[str, float]:
    y_true = pd.Series(y_true).astype(float)
    y_pred = pd.Series(y_pred).astype(float)
    nonzero = y_true.replace(0, np.nan)
    mape = (np.abs((y_true - y_pred) / nonzero)).replace([np.inf, -np.inf], np.nan).mean()
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(mean_squared_error(y_true, y_pred))),
        "mape": float(mape) if not np.isnan(mape) else np.nan,
    }


def make_lag_frame(series: pd.Series) -> pd.DataFrame:
    frame = pd.DataFrame({"y": series})
    for lag in [1, 2, 3, 7, 14]:
        frame[f"lag_{lag}"] = frame["y"].shift(lag)
    for window in [3, 7, 14]:
        frame[f"roll_mean_{window}"] = frame["y"].shift(1).rolling(window).mean()
        frame[f"roll_std_{window}"] = frame["y"].shift(1).rolling(window).std()
    frame["weekday"] = frame.index.weekday
    frame["is_weekend"] = frame.index.weekday.isin([5, 6]).astype(int)
    frame["month"] = frame.index.month
    frame["day"] = frame.index.day
    frame["is_month_start_period"] = (frame.index.day <= 10).astype(int)
    frame["is_month_end_period"] = (frame.index.day >= 21).astype(int)
    return frame.dropna()


def one_step_features(history: pd.Series, pred_date: pd.Timestamp, feature_cols: list[str]) -> pd.DataFrame:
    values: dict[str, float] = {}
    for lag in [1, 2, 3, 7, 14]:
        values[f"lag_{lag}"] = float(history.iloc[-lag]) if len(history) >= lag else float(history.mean())
    for window in [3, 7, 14]:
        tail = history.iloc[-window:]
        values[f"roll_mean_{window}"] = float(tail.mean())
        values[f"roll_std_{window}"] = float(tail.std(ddof=1)) if len(tail) > 1 else 0.0
    values["weekday"] = pred_date.weekday()
    values["is_weekend"] = int(pred_date.weekday() in [5, 6])
    values["month"] = pred_date.month
    values["day"] = pred_date.day
    values["is_month_start_period"] = int(pred_date.day <= 10)
    values["is_month_end_period"] = int(pred_date.day >= 21)
    return pd.DataFrame([{col: values[col] for col in feature_cols}])


def recursive_rf_forecast(train: pd.Series, pred_index: pd.DatetimeIndex) -> pd.Series:
    lag_frame = make_lag_frame(train)
    X_train = lag_frame.drop(columns=["y"])
    y_train = lag_frame["y"]
    model = RandomForestRegressor(n_estimators=400, random_state=42, min_samples_leaf=2)
    model.fit(X_train, y_train)

    history = train.copy()
    preds = []
    for pred_date in pred_index:
        x = one_step_features(history, pred_date, list(X_train.columns))
        pred = max(0.0, float(model.predict(x)[0]))
        preds.append(pred)
        history.loc[pred_date] = pred
    return pd.Series(preds, index=pred_index)


def sarima_forecast(train: pd.Series, pred_index: pd.DatetimeIndex) -> pd.Series:
    try:
        train_transformed = np.log1p(train.clip(lower=0))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = SARIMAX(
                train_transformed,
                order=(1, 1, 1),
                seasonal_order=(1, 0, 1, 7),
                enforce_stationarity=False,
                enforce_invertibility=False,
            )
            result = model.fit(disp=False)
            pred = np.expm1(result.get_forecast(steps=len(pred_index)).predicted_mean)
            pred.index = pred_index
            return pred.clip(lower=0)
    except Exception:
        fallback = pd.Series([train.iloc[-7:].mean()] * len(pred_index), index=pred_index)
        return fallback.clip(lower=0)


class LSTMRegressor(nn.Module):
    def __init__(self, hidden_size: int = 24) -> None:
        super().__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=1, batch_first=True)
        self.linear = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.lstm(x)
        return self.linear(output[:, -1, :])


def make_lstm_training_tensors(train: pd.Series, sequence_length: int) -> tuple[torch.Tensor, torch.Tensor, float, float, list[float]]:
    transformed = np.log1p(train.clip(lower=0).astype(float).to_numpy())
    mean = float(transformed.mean())
    std = float(transformed.std())
    if std == 0:
        std = 1.0
    scaled = ((transformed - mean) / std).astype(np.float32)
    X, y = [], []
    for idx in range(sequence_length, len(scaled)):
        X.append(scaled[idx - sequence_length : idx])
        y.append(scaled[idx])
    X_tensor = torch.tensor(np.array(X), dtype=torch.float32).unsqueeze(-1)
    y_tensor = torch.tensor(np.array(y), dtype=torch.float32).unsqueeze(-1)
    return X_tensor, y_tensor, mean, std, scaled.tolist()


def lstm_forecast(
    train: pd.Series,
    pred_index: pd.DatetimeIndex,
    sequence_length: int = 14,
    epochs: int = 260,
) -> pd.Series:
    if len(train) <= sequence_length + 5:
        return pd.Series([train.iloc[-7:].mean()] * len(pred_index), index=pred_index).clip(lower=0)

    torch.manual_seed(42)
    X_train, y_train, mean, std, history_scaled = make_lstm_training_tensors(train, sequence_length)
    model = LSTMRegressor(hidden_size=24)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

    model.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        loss.backward()
        optimizer.step()

    model.eval()
    preds = []
    with torch.no_grad():
        for pred_date in pred_index:
            x = torch.tensor(history_scaled[-sequence_length:], dtype=torch.float32).reshape(1, sequence_length, 1)
            pred_scaled = float(model(x).item())
            history_scaled.append(pred_scaled)
            pred_log = pred_scaled * std + mean
            preds.append(max(0.0, float(np.expm1(pred_log))))
    return pd.Series(preds, index=pred_index)


def forecast_one_target(series: pd.Series, target_name: str, test_size: int = 14, future_days: int = 14) -> tuple[pd.DataFrame, pd.DataFrame]:
    series = series.asfreq("D").astype(float)
    train = series.iloc[:-test_size]
    test = series.iloc[-test_size:]

    sarima_test = sarima_forecast(train, test.index)
    rf_test = recursive_rf_forecast(train, test.index)
    lstm_test = lstm_forecast(train, test.index)
    ensemble_test = (sarima_test + rf_test + lstm_test) / 3

    metric_rows = []
    for model_name, pred in [
        ("SARIMA", sarima_test),
        ("RandomForest_lag", rf_test),
        ("LSTM", lstm_test),
        ("Ensemble_mean", ensemble_test),
    ]:
        metric_rows.append({"target": target_name, "model": model_name, **forecast_metrics(test, pred)})

    future_index = pd.date_range(series.index.max() + pd.Timedelta(days=1), periods=future_days, freq="D")
    sarima_future = sarima_forecast(series, future_index)
    rf_future = recursive_rf_forecast(series, future_index)
    lstm_future = lstm_forecast(series, future_index)
    ensemble_future = (sarima_future + rf_future + lstm_future) / 3
    future = pd.DataFrame(
        {
            "date": future_index,
            "target": target_name,
            "sarima": sarima_future.values,
            "random_forest_lag": rf_future.values,
            "lstm": lstm_future.values,
            "ensemble_mean": ensemble_future.values,
        }
    )

    test_pred = pd.DataFrame(
        {
            "date": test.index,
            "target": target_name,
            "actual": test.values,
            "sarima": sarima_test.values,
            "random_forest_lag": rf_test.values,
            "lstm": lstm_test.values,
            "ensemble_mean": ensemble_test.values,
        }
    )
    preds = pd.concat([test_pred.assign(period="test"), future.assign(actual=np.nan, period="future")], ignore_index=True)
    return pd.DataFrame(metric_rows), preds


def run_forecasting(daily: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    daily_indexed = daily.set_index("date")
    metric_frames = []
    pred_frames = []
    for target in ["total_amount", "transaction_count"]:
        metrics, preds = forecast_one_target(daily_indexed[target], target_name=target)
        metric_frames.append(metrics)
        pred_frames.append(preds)
    return pd.concat(metric_frames, ignore_index=True), pd.concat(pred_frames, ignore_index=True)


def save_plots(
    survey: pd.DataFrame,
    student_features: pd.DataFrame,
    cluster_eval: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    X_scaled: np.ndarray,
    daily: pd.DataFrame,
    forecast_preds: pd.DataFrame,
) -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    ELBOW.mkdir(parents=True, exist_ok=True)
    elbow_k = select_elbow_k(cluster_eval)
    chosen_k = int(cluster_profile["cluster"].nunique())

    daily_plot = daily.copy()
    daily_plot["rolling_7d_amount"] = daily_plot["total_amount"].rolling(7, min_periods=1).mean()
    plt.figure(figsize=(12, 5))
    plt.plot(daily_plot["date"], daily_plot["total_amount"], label="日消费金额", linewidth=1.2)
    plt.plot(daily_plot["date"], daily_plot["rolling_7d_amount"], label="7日滚动均值", linewidth=2.2)
    plt.title("日度消费金额趋势")
    plt.xlabel("日期")
    plt.ylabel("金额")
    plt.legend()
    plt.tight_layout()
    plt.savefig(FIG / "daily_total_amount_trend.png", dpi=180)
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(cluster_eval["k"], cluster_eval["inertia"], marker="o", linewidth=2.2)
    plt.axvline(elbow_k, color="#c44e52", linestyle="--", linewidth=1.8, label=f"肘部点 k={elbow_k}")
    plt.title("KMeans 聚类数选择：SSE 肘部法则")
    plt.xlabel("聚类数 k")
    plt.ylabel("类内平方和 SSE")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ELBOW / "cluster_elbow_inertia.png", dpi=180)
    plt.close()

    drop_df = cluster_eval.dropna(subset=["inertia_drop_rate"]).copy()
    plt.figure(figsize=(9, 5))
    sns.barplot(data=drop_df, x="k", y="inertia_drop_rate", color="#4c72b0")
    plt.axvline(drop_df["k"].astype(str).tolist().index(str(elbow_k)), color="#c44e52", linestyle="--", linewidth=1.8)
    plt.title("KMeans 聚类数选择：SSE 边际下降率")
    plt.xlabel("聚类数 k")
    plt.ylabel("相对上一 k 的 SSE 下降率")
    plt.gca().yaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
    plt.tight_layout()
    plt.savefig(ELBOW / "cluster_inertia_drop_rate.png", dpi=180)
    plt.close()

    best_silhouette_k = int(cluster_eval.sort_values("silhouette", ascending=False).iloc[0]["k"])
    plt.figure(figsize=(9, 5))
    plt.plot(cluster_eval["k"], cluster_eval["silhouette"], marker="o", linewidth=2.2)
    plt.axvline(chosen_k, color="#c44e52", linestyle="--", linewidth=1.8, label=f"最终 k={chosen_k}")
    plt.axvline(best_silhouette_k, color="#55a868", linestyle=":", linewidth=1.8, label=f"轮廓系数最高 k={best_silhouette_k}")
    plt.title("KMeans 聚类数选择：轮廓系数")
    plt.xlabel("聚类数 k")
    plt.ylabel("Silhouette Score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(ELBOW / "cluster_silhouette_score.png", dpi=180)
    plt.close()

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].plot(cluster_eval["k"], cluster_eval["calinski_harabasz"], marker="o", linewidth=2.2)
    axes[0].axvline(chosen_k, color="#c44e52", linestyle="--", linewidth=1.8)
    axes[0].set_title("Calinski-Harabasz 指数")
    axes[0].set_xlabel("聚类数 k")
    axes[0].set_ylabel("CH")
    axes[1].plot(cluster_eval["k"], cluster_eval["davies_bouldin"], marker="o", linewidth=2.2)
    axes[1].axvline(chosen_k, color="#c44e52", linestyle="--", linewidth=1.8)
    axes[1].set_title("Davies-Bouldin 指数")
    axes[1].set_xlabel("聚类数 k")
    axes[1].set_ylabel("DB")
    plt.suptitle("KMeans 聚类数选择：辅助有效性指标")
    plt.tight_layout()
    plt.savefig(ELBOW / "cluster_validity_indices.png", dpi=180)
    plt.close()

    pca_for_candidates = PCA(n_components=2, random_state=42)
    candidate_coords = pca_for_candidates.fit_transform(X_scaled)
    for k in [2, 3, 4]:
        if k >= len(student_features):
            continue
        model = KMeans(n_clusters=k, init="k-means++", n_init=50, random_state=42)
        labels = model.fit_predict(X_scaled)
        candidate = student_features.copy()
        candidate["cluster"] = labels
        candidate_profile = candidate.groupby("cluster")[["total_amount"]].mean()
        candidate_profile["cluster_name"] = candidate_profile.index.map(name_clusters(candidate_profile))
        candidate["cluster_name"] = candidate["cluster"].map(candidate_profile["cluster_name"])

        pca_df = pd.DataFrame(candidate_coords, columns=["PC1", "PC2"])
        pca_df["cluster_name"] = candidate["cluster_name"].values
        legend_order = candidate_profile.sort_values("total_amount", ascending=False)["cluster_name"].tolist()

        plt.figure(figsize=(9, 6))
        sns.scatterplot(
            data=pca_df,
            x="PC1",
            y="PC2",
            hue="cluster_name",
            hue_order=legend_order,
            s=28,
            alpha=0.8,
        )
        plt.title(f"KMeans 聚类结果 PCA 投影：k={k}")
        plt.xlabel("PC1")
        plt.ylabel("PC2")
        plt.legend(title="消费类别", bbox_to_anchor=(1.02, 1), loc="upper left")
        plt.tight_layout()
        plt.savefig(ELBOW / f"candidate_k{k}_pca.png", dpi=180)
        plt.close()

    plt.figure(figsize=(9, 5))
    order = cluster_profile.sort_values("total_amount", ascending=False)["cluster_name"]
    sns.barplot(data=cluster_profile, x="cluster_name", y="students", order=order)
    plt.title("消费画像群体人数分布")
    plt.xlabel("画像类型")
    plt.ylabel("人数")
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    plt.savefig(FIG / "cluster_distribution.png", dpi=180)
    plt.close()

    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(coords, columns=["PC1", "PC2"])
    pca_df["cluster_name"] = student_features["cluster_name"].values
    plt.figure(figsize=(9, 6))
    sns.scatterplot(data=pca_df, x="PC1", y="PC2", hue="cluster_name", s=28, alpha=0.8)
    plt.title("KMeans++ 消费画像聚类 PCA 投影")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.legend(title="画像类型", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(FIG / "cluster_pca.png", dpi=180)
    plt.close()

    meal_cols = [
        "meal_amount_breakfast_share",
        "meal_amount_lunch_share",
        "meal_amount_dinner_share",
        "meal_amount_late_night_share",
    ]
    meal_profile = cluster_profile[["cluster_name", *meal_cols]].melt("cluster_name", var_name="meal", value_name="share")
    meal_profile["meal"] = meal_profile["meal"].map(
        {
            "meal_amount_breakfast_share": "早餐",
            "meal_amount_lunch_share": "午餐",
            "meal_amount_dinner_share": "晚餐",
            "meal_amount_late_night_share": "宵夜",
        }
    )
    plt.figure(figsize=(10, 5))
    sns.barplot(data=meal_profile, x="cluster_name", y="share", hue="meal", order=list(order))
    plt.title("各画像群体餐次金额占比")
    plt.xlabel("画像类型")
    plt.ylabel("金额占比")
    plt.xticks(rotation=0, ha="center")
    plt.tight_layout()
    plt.savefig(FIG / "cluster_meal_share.png", dpi=180)
    plt.close()

    heatmap_cols = [
        "total_amount",
        "txn_count",
        "avg_amount",
        "active_days",
        "merchant_count",
        "pay_amount_scan_pay_share",
        "weekend_amount_share",
        "merchant_amount_fruit_share",
    ]
    heatmap_data = cluster_profile.sort_values("total_amount", ascending=False).set_index("cluster_name")[heatmap_cols]
    heatmap_data = heatmap_data.rename(
        columns={
            "total_amount": "总消费",
            "txn_count": "交易笔数",
            "avg_amount": "单笔均值",
            "active_days": "活跃天数",
            "merchant_count": "商户数",
            "pay_amount_scan_pay_share": "扫码占比",
            "weekend_amount_share": "周末占比",
            "merchant_amount_fruit_share": "水果占比",
        }
    )
    heatmap_scaled = (heatmap_data - heatmap_data.mean()) / heatmap_data.std(ddof=0).replace(0, np.nan)
    heatmap_scaled = heatmap_scaled.fillna(0)
    plt.figure(figsize=(11, 5))
    sns.heatmap(heatmap_scaled, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.5)
    plt.title("最终聚类中心特征热力图")
    plt.xlabel("特征")
    plt.ylabel("消费类别")
    plt.xticks(rotation=0, ha="center")
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(FIG / "cluster_center_heatmap.png", dpi=180)
    plt.close()

    q13 = survey[survey["question_id"].eq(13)].copy()
    if not q13.empty:
        q13 = q13.sort_values("percent", ascending=False)
        plt.figure(figsize=(10, 5))
        sns.barplot(data=q13, y="option", x="percent")
        plt.title("问卷：月消费占比最高项")
        plt.xlabel("选择比例")
        plt.ylabel("")
        plt.gca().xaxis.set_major_formatter(lambda x, pos: f"{x:.0%}")
        plt.tight_layout()
        plt.savefig(FIG / "questionnaire_top_consumption_items.png", dpi=180)
        plt.close()

    forecast_plot_meta = {
        "total_amount": ("消费金额预测效果：测试集最后14天", "金额", "forecast_total_amount_test.png"),
        "transaction_count": ("交易笔数预测效果：测试集最后14天", "笔数", "forecast_transaction_count_test.png"),
    }
    for target, (title, ylabel, filename) in forecast_plot_meta.items():
        test_target = forecast_preds[(forecast_preds["target"] == target) & (forecast_preds["period"] == "test")]
        plt.figure(figsize=(12, 5))
        plt.plot(test_target["date"], test_target["actual"], marker="o", label="实际值")
        plt.plot(test_target["date"], test_target["sarima"], marker="o", label="SARIMA")
        plt.plot(test_target["date"], test_target["random_forest_lag"], marker="o", label="随机森林滞后特征")
        plt.plot(test_target["date"], test_target["lstm"], marker="o", label="LSTM")
        plt.plot(test_target["date"], test_target["ensemble_mean"], marker="o", label="融合模型")
        plt.title(title)
        plt.xlabel("日期")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / filename, dpi=180)
        plt.close()

    lstm_plot_meta = {
        "total_amount": ("LSTM 消费金额预测效果：测试集最后14天", "金额", "forecast_lstm_total_amount_test.png"),
        "transaction_count": ("LSTM 交易笔数预测效果：测试集最后14天", "笔数", "forecast_lstm_transaction_count_test.png"),
    }
    for target, (title, ylabel, filename) in lstm_plot_meta.items():
        test_target = forecast_preds[(forecast_preds["target"] == target) & (forecast_preds["period"] == "test")]
        plt.figure(figsize=(12, 5))
        plt.plot(test_target["date"], test_target["actual"], marker="o", label="实际值", linewidth=2.2)
        plt.plot(test_target["date"], test_target["lstm"], marker="o", label="LSTM预测值", linewidth=2.2)
        plt.title(title)
        plt.xlabel("日期")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / filename, dpi=180)
        plt.close()

    rf_plot_meta = {
        "total_amount": ("随机森林消费金额预测效果：测试集最后14天", "金额", "forecast_random_forest_total_amount_test.png"),
        "transaction_count": ("随机森林交易笔数预测效果：测试集最后14天", "笔数", "forecast_random_forest_transaction_count_test.png"),
    }
    for target, (title, ylabel, filename) in rf_plot_meta.items():
        test_target = forecast_preds[(forecast_preds["target"] == target) & (forecast_preds["period"] == "test")]
        plt.figure(figsize=(12, 5))
        plt.plot(test_target["date"], test_target["actual"], marker="o", label="实际值", linewidth=2.2)
        plt.plot(test_target["date"], test_target["random_forest_lag"], marker="o", label="随机森林预测值", linewidth=2.2)
        plt.title(title)
        plt.xlabel("日期")
        plt.ylabel(ylabel)
        plt.legend()
        plt.tight_layout()
        plt.savefig(FIG / filename, dpi=180)
        plt.close()


def save_outputs(
    survey: pd.DataFrame,
    student_features: pd.DataFrame,
    cluster_eval: pd.DataFrame,
    cluster_profile: pd.DataFrame,
    daily: pd.DataFrame,
    forecast_metrics_df: pd.DataFrame,
    forecast_preds: pd.DataFrame,
) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ELBOW.mkdir(parents=True, exist_ok=True)
    safe_features = student_features.drop(columns=["first_date", "last_date"], errors="ignore")
    safe_features.to_csv(OUT / "student_features_with_clusters.csv", index=False, encoding="utf-8-sig")
    cluster_profile.to_csv(OUT / "cluster_profile.csv", index=False, encoding="utf-8-sig")
    daily.to_csv(OUT / "daily_trend.csv", index=False, encoding="utf-8-sig")
    forecast_metrics_df.to_csv(OUT / "forecast_metrics.csv", index=False, encoding="utf-8-sig")
    forecast_preds.to_csv(OUT / "daily_forecast.csv", index=False, encoding="utf-8-sig")
    cluster_eval.to_csv(ELBOW / "cluster_evaluation.csv", index=False, encoding="utf-8-sig")


def main() -> None:
    setup_plot_style()
    OUT.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)

    survey = read_extracted_questionnaire()
    consumption, meta = read_consumption()
    student_features = build_student_features(consumption)

    cluster_features, cluster_eval, cluster_profile, X_scaled = cluster_students(student_features)
    daily = build_daily_series(consumption)
    forecast_metrics_df, forecast_preds = run_forecasting(daily)

    save_outputs(
        survey,
        cluster_features,
        cluster_eval,
        cluster_profile,
        daily,
        forecast_metrics_df,
        forecast_preds,
    )
    save_plots(survey, cluster_features, cluster_eval, cluster_profile, X_scaled, daily, forecast_preds)

    print(f"Done. Results saved to: {OUT}")
    print(f"Students: {meta['students']}, records: {meta['clean_rows']}, date range: {meta['date_min']} to {meta['date_max']}")
    print(f"Chosen k: {cluster_profile['cluster'].nunique()}")


if __name__ == "__main__":
    main()
