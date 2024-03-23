"""Training for Kaggle competition Trading at the Close by Optiver."""

import dataclasses
import gc
import pickle
import warnings
from itertools import combinations
from warnings import simplefilter

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import xgboost as xgb
from numba import njit, prange
from scipy.stats import norm
from sklearn.dummy import DummyRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_absolute_percentage_error

SHIFT_WINDOWS = [1, 2, 3, 6, 10]
SHIFT_GROUP = ["date_id", "stock_id"]

TICK_ANALYSIS_PATH = f"data/"
OUTPUT_PATH = "output"
TRAIN_FILE = f"data/train.csv"

DROP_COLS = ["target", "time_id", "row_id"]

warnings.filterwarnings("ignore")
warnings.simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

DEFAULT_DROP_FEATURES = ["date_id", "row_id", "time_id", "target"]
DEFAULT_PRICE_FEATURES = ["bid_price", "ask_price", "reference_price", "wap"]
TO_CENTER = ['reference_price', 'far_price', 'near_price', 'ask_price', 'bid_price', "mid_spot"]

INDEX_WEIGHTS = [
    0.004, 0.001, 0.002, 0.006, 0.004, 0.004, 0.002, 0.006, 0.006, 0.002, 0.002, 0.008,
    0.006, 0.002, 0.008, 0.006, 0.002, 0.006, 0.004, 0.002, 0.004, 0.001, 0.006, 0.004,
    0.002, 0.002, 0.004, 0.002, 0.004, 0.004, 0.001, 0.001, 0.002, 0.002, 0.006, 0.004,
    0.004, 0.004, 0.006, 0.002, 0.002, 0.04, 0.002, 0.002, 0.004, 0.04, 0.002, 0.001,
    0.006, 0.004, 0.004, 0.006, 0.001, 0.004, 0.004, 0.002, 0.006, 0.004, 0.006, 0.004,
    0.006, 0.004, 0.002, 0.001, 0.002, 0.004, 0.002, 0.008, 0.004, 0.004, 0.002, 0.004,
    0.006, 0.002, 0.004, 0.004, 0.002, 0.004, 0.004, 0.004, 0.001, 0.002, 0.002, 0.008,
    0.02, 0.004, 0.006, 0.002, 0.02, 0.002, 0.002, 0.006, 0.004, 0.002, 0.001, 0.02,
    0.006, 0.001, 0.002, 0.004, 0.001, 0.002, 0.006, 0.006, 0.004, 0.006, 0.001, 0.002,
    0.004, 0.006, 0.006, 0.001, 0.04, 0.006, 0.002, 0.004, 0.002, 0.002, 0.006, 0.002,
    0.002, 0.004, 0.006, 0.006, 0.002, 0.002, 0.008, 0.006, 0.004, 0.002, 0.006, 0.002,
    0.004, 0.006, 0.002, 0.004, 0.001, 0.004, 0.002, 0.004, 0.008, 0.006, 0.008, 0.002,
    0.004, 0.002, 0.001, 0.004, 0.004, 0.004, 0.006, 0.008, 0.004, 0.001, 0.001, 0.002,
    0.006, 0.004, 0.001, 0.002, 0.006, 0.004, 0.006, 0.008, 0.002, 0.002, 0.004, 0.002,
    0.04, 0.002, 0.002, 0.004, 0.002, 0.002, 0.006, 0.02, 0.004, 0.002, 0.006, 0.02,
    0.001, 0.002, 0.006, 0.004, 0.006, 0.004, 0.004, 0.004, 0.004, 0.002, 0.004, 0.04,
    0.002, 0.008, 0.002, 0.004, 0.001, 0.004, 0.006, 0.004,
]  # noqa

INDEX_WEIGHTS = {i: w for i, w in enumerate(INDEX_WEIGHTS)}

# Based on https://www.kaggle.com/code/lognorm/de-anonymizing-stock-id
PROBLEM_TICKS = {8, 29, 34, 37, 39, 57, 58, 112, 118}  # Not a good match during reverse engineering
LEADS = [45, 84, 168, 41, 191]  # MSFT, GOOGL, AMZN, NVDA, TSLA

# Decided during feature selection stage
DROP_FEATURES = ['bid_price_wap_imb',
                 'reference_price_shift_10',
                 'matched_imbalance',
                 'mean_wap_time_bucket_ret_2',
                 'stock_id',
                 'imbalance_size_shift_10',
                 'far_price_wap_imb',
                 'matched_size_shift_3',
                 'liquidity_imbalance',
                 'inferred_price_daily_pct_change']

# #Days to evaluate the full model
HOLDOUT_DAYS = 10

# Number of GBT estimators
N_ESTIMATORS = 6000

# Models to use in inference, selection based on a separate evaluation
SELECTED_MODELS = ['model_dummy.pkl',
                   'model_lgb_fold_4.pkl',
                   'model_lgb_fold_5.pkl',
                   'model_lgb_full.pkl',
                   'model_xgb_fold_2.pkl',
                   'model_xgb_fold_3.pkl',
                   'model_xgb_fold_4.pkl',
                   'model_xgb_fold_5.pkl']


@dataclasses.dataclass
class LookupInfo:
    std_wap: pd.Series
    std_index: pd.Series
    median_index: pd.Series
    median_price: pd.Series
    stock_info: pd.DataFrame
    global_stock_dict: dict

    def save(self, path):
        with open(path, "wb") as fh:
            pickle.dump(self, fh)

    @classmethod
    def load(cls, path):
        with open(path, "wb") as fh:
            return pickle.load(fh)


def reduce_mem_usage(X):
    """
    Iterate through all numeric columns of a dataframe and modify the data type
    to reduce memory usage.
    """

    start_mem = X.memory_usage().sum() / 1024 ** 2
    # print(f'Memory usage of dataframe is {start_mem:.2f} MB')

    for col in X.columns:
        col_type = X[col].dtype

        if col_type != object:
            c_min = X[col].min()
            c_max = X[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    X[col] = X[col].astype(np.int16)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    X[col] = X[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    X[col] = X[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    X[col] = X[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    X[col] = X[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    X[col] = X[col].astype(np.float32)
                else:
                    X[col] = X[col].astype(np.float32)
    end_mem = X.memory_usage().sum() / 1024 ** 2
    #     print(f'Memory usage after optimization is: {end_mem:.2f} MB')
    decrease = 100 * (start_mem - end_mem) / start_mem
    print(f'Decreased memory use by {start_mem - end_mem:.2f} MB')

    return X


# From https://www.kaggle.com/code/judith007/lb-5-3405-rapids-gpu-speeds-up-feature-engineer
@njit(parallel=True)
def compute_triplet_imbalance(df_values, comb_indices):
    num_rows = df_values.shape[0]
    num_combinations = len(comb_indices)
    imbalance_features = np.empty((num_rows, num_combinations))

    for i in prange(num_combinations):
        a, b, c = comb_indices[i]

        for j in range(num_rows):
            max_val = max(df_values[j, a], df_values[j, b], df_values[j, c])
            min_val = min(df_values[j, a], df_values[j, b], df_values[j, c])
            mid_val = df_values[j, a] + df_values[j, b] + df_values[j, c] - min_val - max_val

            if mid_val == min_val:
                imbalance_features[j, i] = np.nan
            else:
                imbalance_features[j, i] = (max_val - mid_val) / (mid_val - min_val)

    return imbalance_features


# https://www.kaggle.com/code/judith007/lb-5-3405-rapids-gpu-speeds-up-feature-engineer
def calculate_triplet_imbalance_numba(price, X):
    """Fast triplet imbalance calculation."""
    # Convert DataFrame to numpy array for Numba compatibility
    df_values = X[price].values
    comb_indices = [(price.index(a), price.index(b), price.index(c)) for a, b, c in combinations(price, 3)]

    # Calculate the triplet imbalance using the Numba-optimized function
    features_array = compute_triplet_imbalance(df_values, comb_indices)

    # Create a DataFrame from the results
    columns = [f"{a}_{b}_{c}_imb2" for a, b, c in combinations(price, 3)]
    features = pd.DataFrame(features_array, columns=columns)

    return features


def load_train(path):
    return pd.read_csv(path).pipe(reduce_mem_usage)


def create_lookup():
    """Create stock lookup data."""
    train = load_train(TRAIN_FILE)
    std_wap = train.groupby('stock_id')["wap"].std()

    global_stock_id_feats = {
        "median_size": train.groupby("stock_id")["bid_size"].median() + train.groupby("stock_id")[
            "ask_size"].median(),
        "std_size": train.groupby("stock_id")["bid_size"].std() + train.groupby("stock_id")["ask_size"].std(),
        "ptp_size": train.groupby("stock_id")["bid_size"].max() - train.groupby("stock_id")["bid_size"].min(),
        "median_price": train.groupby("stock_id")["bid_price"].median() + train.groupby("stock_id")[
            "ask_price"].median(),
        "std_price": train.groupby("stock_id")["bid_price"].std() + train.groupby("stock_id")["ask_price"].std(),
        "ptp_price": train.groupby("stock_id")["bid_price"].max() - train.groupby("stock_id")["ask_price"].min(),
    }

    std_index = train.groupby(["date_id", "seconds_in_bucket"]).wap.mean().std()
    median_index = train.groupby(["date_id", "seconds_in_bucket"]).wap.mean().median()

    delta_bid_price = train[["date_id", "stock_id", "seconds_in_bucket", "bid_price"]].copy()
    delta_bid_price["delta"] = delta_bid_price.groupby(["date_id", "stock_id"],
                                                       sort=False).bid_price.diff().abs().rename(
        "delta")
    delta_bid_price.loc[delta_bid_price.delta == 0, "delta"] = np.nan

    delta_bid_price["price_est"] = 0.01 / delta_bid_price.delta
    delta_bid_price["price_est"] = delta_bid_price.groupby(["date_id", "stock_id"]).price_est.transform("max")

    median_price = delta_bid_price.groupby("stock_id").price_est.median()

    # Based on https://www.kaggle.com/code/lognorm/de-anonymizing-stock-id + historical open/close/high/low data
    stock_info = pd.read_csv(f"{TICK_ANALYSIS_PATH}/stock_info.csv", index_col=0)
    stock_info.head()

    stock_info["close_coevar"] = stock_info.close_std / stock_info.close_mean

    return LookupInfo(std_wap=std_wap,
                      std_index=std_index, median_index=median_index, median_price=median_price,
                      stock_info=stock_info,
                      global_stock_dict=global_stock_id_feats)


def imbalance_features(X, **kwargs):
    """Imbalance features."""
    prices = ["reference_price", "far_price", "near_price", "ask_price", "bid_price", "wap"]
    sizes = ["matched_size", "bid_size", "ask_size", "imbalance_size"]
    X["volume"] = X.eval("ask_size + bid_size")
    X["mid_price"] = X.eval("(ask_price + bid_price) / 2")
    X["liquidity_imbalance"] = X.eval("(bid_size-ask_size)/(bid_size+ask_size)")
    X["matched_imbalance"] = X.eval("(imbalance_size-matched_size)/(matched_size+imbalance_size)")
    X["size_imbalance"] = X.eval("bid_size / ask_size")

    for c in combinations(prices, 2):
        X[f"{c[0]}_{c[1]}_imb"] = X.eval(f"({c[0]} - {c[1]})/({c[0]} + {c[1]})")

    for c in [['ask_price', 'bid_price', 'wap', 'reference_price'], sizes]:
        triplet_feature = calculate_triplet_imbalance_numba(c, X)
        X[triplet_feature.columns] = triplet_feature.values

    X["imbalance_momentum"] = X.groupby(['stock_id'])['imbalance_size'].diff(periods=1) / X['matched_size']
    X["price_spread"] = X["ask_price"] - X["bid_price"]
    X["spread_intensity"] = X.groupby(['stock_id'])['price_spread'].diff()
    X['price_pressure'] = X['imbalance_size'] * (X['ask_price'] - X['bid_price'])
    X['market_urgency'] = X['price_spread'] * X['liquidity_imbalance']
    X['depth_pressure'] = (X['ask_size'] - X['bid_size']) * (X['far_price'] - X['near_price'])

    for func in ["mean", "std", "skew", "kurt"]:
        X[f"all_prices_{func}"] = X[prices].agg(func, axis=1)
        X[f"all_sizes_{func}"] = X[sizes].agg(func, axis=1)

    for col in ['matched_size', 'imbalance_size', 'reference_price', 'imbalance_buy_sell_flag']:
        for window in SHIFT_WINDOWS:
            X[f"{col}_shift_{window}"] = X.groupby(SHIFT_GROUP)[col].shift(window)

    for col in ['matched_size', 'imbalance_size', 'reference_price']:
        for window in SHIFT_WINDOWS:
            X[f"{col}_ret_{window}"] = X.groupby(SHIFT_GROUP)[col].pct_change(window)

    for col in ['ask_price', 'bid_price', 'ask_size', 'bid_size', 'market_urgency', 'imbalance_momentum',
                'size_imbalance']:
        for window in SHIFT_WINDOWS:
            X[f"{col}_diff_{window}"] = X.groupby(SHIFT_GROUP)[col].diff(window)

    return X.replace([np.inf, -np.inf], 0)


def make_revealed_targets(X):
    """Used to create revealed targets during training."""
    X = X[["date_id", "seconds_in_bucket", "stock_id", "target"]].copy()
    X = X.rename(columns={"target": "revealed_target"})
    X["revealed_date_id"] = X["date_id"]
    X["date_id"] += 1
    return X


def other_features(X, lookup: LookupInfo, online=False, **kwargs):
    """All other features go here."""
    # Time
    X["seconds"] = X["seconds_in_bucket"] % 60
    X["minute"] = X["seconds_in_bucket"] // 60
    for key, value in lookup.global_stock_dict.items():
        X[f"global_{key}"] = X["stock_id"].map(value.to_dict())

    # Based on reverse eng. stock IDs
    stock_info_cols = ["stock_id", "sector_id", "daily_move_pct_mean", "in_nasdaq_100_first"]  # , "in_nasdaq_100_first"
    stock_info_view = lookup.stock_info[stock_info_cols]  # ,"open"
    X = X.merge(stock_info_view, on=["stock_id"], how="left", validate="many_to_one")
    X["problem_tick"] = X.stock_id.isin(PROBLEM_TICKS)

    # Group features
    X["mean_wap_time_bucket"] = X.groupby(["date_id", "seconds_in_bucket"]).wap.transform("mean")
    X["rank_wap_time_bucket"] = X.groupby(["date_id", "seconds_in_bucket"]).wap.transform("rank")
    total_imbalance_time_bucket = X.groupby(["date_id", "seconds_in_bucket"]).imbalance_size.transform("sum")

    X["signed_imbalance_size"] = (X["imbalance_buy_sell_flag"] * X[
        "imbalance_size"]) / total_imbalance_time_bucket  # np.log(X["imbalance_size"]+1)

    X["sector_index"] = X.groupby(["date_id", "seconds_in_bucket", "sector_id"]).wap.transform("mean")
    X["sector_imbalance"] = X.groupby(["date_id", "seconds_in_bucket", "sector_id"]).imbalance_buy_sell_flag.transform(
        "mean")

    X["performance_over_index"] = 10000 * (X["wap"] - X["mean_wap_time_bucket"])
    X["performance_over_sector"] = 10000 * (X["wap"] - X["sector_index"])

    X["normalized_wap_move"] = (1 - X["wap"]) / X["daily_move_pct_mean"]

    X["mean_flag_time_bucket"] = X.groupby(["date_id", "seconds_in_bucket"]).imbalance_buy_sell_flag.transform("mean")

    # "Standard error" of the wap based on the #observations per day and historical volatility
    X_std_wap = X['stock_id'].map(lookup.std_wap.to_dict())
    wap_err = X_std_wap / np.sqrt(1 + X["seconds_in_bucket"] / 10)
    X["wap_err"] = wap_err

    # Previous day features
    revealed_wap = X.groupby("stock_id").wap.shift(55)
    revealed_imbalance = X.groupby("stock_id").imbalance_buy_sell_flag.shift(55)
    revealed_far_price_near_price_imb = X.groupby("stock_id").far_price_near_price_imb.shift(55)

    X["revealed_wap_times_target"] = revealed_wap * X["revealed_target"]
    X["revealed_imbalance_times_target"] = revealed_imbalance * X["revealed_target"]
    X["revealed_far_price_near_price_imb_times_target"] = revealed_far_price_near_price_imb * X["revealed_target"]

    # Based on tick size / inferred price
    price_window = 55
    X["price_movement"] = X["bid_price_diff_1"].abs()
    X.loc[X["price_movement"] == 0, "price_movement"] = np.inf
    tick = X.groupby(["stock_id"], sort=False).rolling(window=price_window, min_periods=1,
                                                       closed="left").price_movement.min().rename("tick").reset_index(
        level=["stock_id"], drop=True)

    inferred_price = (0.01 / tick).rename("inferred_price")
    X["inferred_price"] = inferred_price
    X["inferred_bid_price"] = X["inferred_price"] * X["bid_price"]
    X["inferred_ask_price"] = X["inferred_price"] * X["ask_price"]
    X["inferred_wap"] = X["inferred_price"] * X["wap"]
    X["inferred_bid_volume"] = X["bid_size"] / X["inferred_price"]
    X["inferred_ask_volume"] = X["ask_size"] / X["inferred_price"]
    X["inferred_bid_minus_ask_volume"] = X["inferred_bid_volume"] - X["inferred_ask_volume"]
    X["inferred_imbalance"] = X["imbalance_size"] / X["inferred_price"]

    X["inferred_price_daily_change"] = X.groupby(["stock_id"], sort=False).inferred_price.diff(55).fillna(0)
    del X["price_movement"]
    X["signed_imbalance_volume"] = X["imbalance_buy_sell_flag"] * X["inferred_imbalance"]

    # Diff / Shift/ % Change
    X["stock_weight_index"] = X.stock_id.map(INDEX_WEIGHTS).fillna(0)
    for col in ['mean_wap_time_bucket', 'wap']:
        for window in SHIFT_WINDOWS:
            X[f"{col}_ret_{window}"] = X.groupby(SHIFT_GROUP)[col].pct_change(window)

    for col in ['signed_imbalance_size']:
        for window in SHIFT_WINDOWS:
            X[f"{col}_diff_{window}"] = X.groupby(SHIFT_GROUP)[col].diff(window)

    for window in SHIFT_WINDOWS:
        wap_change = X[f"wap_ret_{window}"]
        X[f"weighted_wap_change_{window}"] = X["stock_weight_index"] * (wap_change + 1)
        X[f"index_ret_{window}"] = X.groupby(["date_id", "seconds_in_bucket"])[
            f"weighted_wap_change_{window}"].transform("sum")
        del X[f"weighted_wap_change_{window}"]
        wap_return = X["wap"] / (X["wap"] - wap_change)
        index_return = X[f"index_ret_{window}"]

        X[f"performance_{window}"] = 10000 * (wap_return - index_return)

    # Rolling window features
    rolling_window = 55
    rolling_group = ["date_id", "stock_id"]

    rolling_stats = {
        np.nanmean: ["imbalance_buy_sell_flag"],
        np.nanmedian: ["performance_over_index", "revealed_target"],
    }
    for func, cols in rolling_stats.items():
        if not online:
            for c in cols:
                name = func.__name__
                col = f"rolling_{name}_{c}_{rolling_window}"
                rolled = X.groupby(rolling_group, sort=False).rolling(window=rolling_window, min_periods=1)[c].apply(
                    func, raw=True, engine="numba").rename(col).reset_index(level=rolling_group, drop=True)
                X[col] = rolled

        else:
            rolling_subset = X.groupby(rolling_group).tail(rolling_window)
            new_cols = []
            for c in cols:
                name = func.__name__
                col = f"rolling_{name}_{c}_{rolling_window}"
                rolling_subset[col] = rolling_subset.groupby(rolling_group)[c].transform(func)
                new_cols.append(col)
            join_cols = ["date_id", "stock_id", "seconds_in_bucket"]
            X = X.merge(rolling_subset[join_cols + new_cols], on=join_cols, how="left")

    # Volatility features
    # Compute at the money call price based on Black-Scholes model
    # S: current price, K: strike price, r: interest, T:time to expiration sigma: yearly stdev returns
    interest = 0.04
    time_to_expiry = (600 - X["seconds_in_bucket"]) / 600

    volatility = X_std_wap  # X["rolling_std_wap_55"]
    d1 = 0 + ((interest + ((volatility ** 2) / 2) * time_to_expiry) / (volatility * np.sqrt(time_to_expiry)))
    d2 = d1 - volatility * np.sqrt(time_to_expiry)
    call_price = X["wap"] * norm.cdf(d1) - X["wap"] * np.exp(-interest * time_to_expiry) * norm.cdf(d2)
    X["call_price_0"] = call_price

    del X["revealed_target"]

    # Features from 'lead stocks' such as MSFT, GOOGL, etc.
    lead_cols = ["wap_ret_1", "imbalance_buy_sell_flag", "call_price_0"]  # , "signed_imbalance_size_diff_1"]
    lead_X = X[X.stock_id.isin(set(LEADS))]
    lead_X = pd.pivot_table(lead_X, index=["date_id", "seconds_in_bucket"],
                            values=lead_cols,
                            # columns=["stock_id"]
                            )

    lead_X = flatten_columns(lead_X, prefix="lead")

    X = X.merge(lead_X, left_on=["date_id", "seconds_in_bucket"],
                right_index=True,
                how="left"
                )

    return X


def flatten_columns(X, prefix=""):
    """Monkey patchable function onto pandas dataframes to flatten MultiIndex column names.

    pd.DataFrame.flatten_columns = flatten_columns
    """
    X.columns = [
        '_'.join([prefix] + [str(x)
                             for x in [y for y in item
                                       if y]]) if not isinstance(item, str) else item
        for item in X.columns
    ]
    return X


def feature_eng(X, lookup: LookupInfo, revealed_targets: pd.DataFrame, **kwargs):
    """Feature engineering."""
    # Select relevant columns for feature generation
    cols = [c for c in X.columns if c not in ["row_id", "time_id", "target", "currently_scored"]]
    X = X[cols]
    X = X.merge(revealed_targets[["date_id", "seconds_in_bucket", "stock_id", "revealed_target"]],
                on=["date_id", "seconds_in_bucket", "stock_id"], how="left")

    # Generate imbalance features
    X = imbalance_features(X, **kwargs)
    X = other_features(X, lookup, **kwargs)
    gc.collect()
    feature_name = [i for i in X.columns if i not in ["row_id", "target", "time_id", "date_id", "currently_scored"]]

    return X[feature_name]


def live_cache(X, history, window):
    if len(history) > 0:
        X = pd.concat(history[-(window):] + [X], axis="rows", ignore_index=True)
    return X


# https://www.mlfinlab.com/en/latest/cross_validation/purged_embargo.html
def group_split(groups, nfolds, gap=2, embargo=2):
    """N-fold cross validation (split by days) with gaps in-between the folds."""
    unique_g = np.sort(np.unique(groups))
    fold_size = (len(unique_g) // nfolds + 1)

    for i in range(nfolds):
        i = nfolds - i - 1
        v0_gap = gap if i > 0 else 0
        v0 = i * fold_size + v0_gap

        v1_gap = gap + embargo if i < nfolds - 1 else 0
        v1 = min(len(unique_g), v0 + fold_size) - v1_gap
        valid_groups = unique_g[v0:v1]
        min_val = min(valid_groups)
        max_val = max(valid_groups)
        train_groups = [g for g in unique_g if
                        g not in valid_groups and np.abs(g - max_val) > gap + embargo and np.abs(g - min_val) > gap]
        valid_idx = groups.isin(valid_groups)
        train_idx = groups.isin(train_groups)
        yield train_idx, valid_idx


def train():
    """Train all models."""
    warnings.filterwarnings("ignore")
    simplefilter(action="ignore", category=pd.errors.PerformanceWarning)

    lookup = create_lookup()
    train = pd.read_csv(TRAIN_FILE).pipe(reduce_mem_usage)
    revealed_targets = make_revealed_targets(train)

    # Test prev input
    X = feature_eng(train, lookup, revealed_targets)
    y = train.target.values
    date_ids = train.date_id

    # This will get replaced by an online model during inference
    dummy_reg = DummyRegressor(strategy="constant", constant=0)
    joblib.dump(dummy_reg, "output/model_dummy.pkl")

    train_xgb(X, y, date_ids)
    train_lgb(X, y, date_ids)


def train_lgb(X, y, date_ids):
    """Train LightGBM models."""
    # Cross-validation parameters
    num_splits = 5
    gap = 4

    holdout_date_id = date_ids.min() + HOLDOUT_DAYS
    holdout_idx = (0 < date_ids) & (date_ids <= holdout_date_id)
    model_idx = date_ids > holdout_date_id
    gc.collect()
    X_model = X[model_idx].drop(columns=DROP_FEATURES, errors="ignore")
    y_model = y[model_idx]
    date_ids_model = date_ids[model_idx]
    X_holdout = X[holdout_idx].drop(columns=DROP_FEATURES, errors="ignore")

    categorical_features = ["stock_group", "sector_id"]
    lgbm_params = {
        "objective": "mae",
        "metric": "mae",
        "n_estimators": N_ESTIMATORS,
        "num_leaves": 256,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        "learning_rate": 0.00871,
        'max_depth': 11,
        "device": "gpu",
        "verbosity": -1,
        "importance_type": "gain",
    }

    mae_scores_kfold = []
    mape_scores_kfold = []
    for fold, (train_idx, valid_idx) in enumerate(group_split(date_ids_model, 5, gap)):
        assert not (date_ids_model[valid_idx].isin(date_ids_model[train_idx]).any())
    for fold, (train_idx, valid_idx) in enumerate(group_split(date_ids_model, 5, gap)):
        X_train, X_valid = X_model[train_idx], X_model[valid_idx]
        y_train, y_valid = y_model[train_idx], y_model[valid_idx]

        print(f"{len(X_train)=} {len(X_valid)=}")

        max_bins = 127
        train_data = lgb.Dataset(X_train, label=y_train,
                                 params=dict(max_bins=max_bins))
        valid_data = lgb.Dataset(X_valid, label=y_valid,
                                 params=dict(max_bins=max_bins))

        valid_sets = [valid_data]
        m = lgb.train(lgbm_params, train_data, valid_sets=valid_sets,
                      categorical_feature=[c for c in categorical_features if c in X_train.columns],
                      callbacks=[lgb.callback.early_stopping(stopping_rounds=100),
                                 lgb.callback.log_evaluation(period=100)])
        print(f"Fold {fold + 1} Training finished.")

        model_filename = f"{OUTPUT_PATH}/model_lgb_fold_{fold + 1}.pkl"
        joblib.dump(m, model_filename)
        y_pred_valid = m.predict(X_valid)

        y_pred_valid = np.nan_to_num(y_pred_valid)
        y_valid = np.nan_to_num(y_valid)

        mae1 = mean_absolute_error(y_valid, y_pred_valid)
        mape1 = mean_absolute_percentage_error(y_valid, y_pred_valid)

        mae_scores_kfold.append(mae1)
        mape_scores_kfold.append(mape1)

    np.mean(mae_scores_kfold)
    average_mae_kfold = np.mean(mae_scores_kfold)
    print(f"{average_mae_kfold=}")

    lgb_models = {}
    for fold in range(0, num_splits):
        model_filename = f"{OUTPUT_PATH}/model_lgb_fold_{fold + 1}.pkl"
        m = joblib.load(model_filename)
        lgb_models[fold] = m

    average_best_iteration = int(np.mean([model.best_iteration for model in lgb_models.values()]))
    print(f"{average_best_iteration=}")
    final_params = lgbm_params.copy()
    final_params["n_estimators"] = average_best_iteration
    average_mae_kfold = np.mean(mae_scores_kfold)
    print(f"{average_mae_kfold=}")

    max_bins = 127
    train_data = lgb.Dataset(X_model, label=y[model_idx], params=dict(max_bins=max_bins))
    valid_data = lgb.Dataset(X_holdout, label=y[holdout_idx], params=dict(max_bins=max_bins))
    m = lgb.train(final_params, train_data, valid_sets=[train_data, valid_data],
                  categorical_feature=[c for c in categorical_features if c in X_model.columns],
                  callbacks=[lgb.callback.log_evaluation(period=100), lgb.callback.early_stopping(100)])
    print(f"Final model training finished.")
    model_filename = f"{OUTPUT_PATH}/model_lgb_full.pkl"
    joblib.dump(m, model_filename)


def train_xgb(X, y, date_ids):
    """Train XGBoost models."""
    NUM_SPLITS = 5
    GAP = 4

    file_names = []
    xgb_params = {
        'learning_rate': 0.008,
        'max_depth': 12,
        'n_estimators': N_ESTIMATORS,
        'max_leaves': 256,
        'objective': 'reg:quantileerror',
        'eval_metric': mean_absolute_error,
        'random_state': 1453,
        'reg_alpha': 1e-6,
        'reg_lambda': 1e-6,
        'verbose': -1,
        "subsample": 0.6,
        "colsample_bytree": 0.8,
        "tree_method": "approx",
        "max_bin": 255,
        "quantile_alpha": .5

    }

    mae_scores_kfold = []
    holdout_date_id = date_ids.min() + HOLDOUT_DAYS
    model_idx = (date_ids > holdout_date_id) & ~(np.isnan(y))
    gc.collect()
    X_model = X[model_idx].drop(columns=DROP_FEATURES, errors="ignore")
    y_model = y[model_idx]
    date_ids_model = date_ids[model_idx]
    for fold, (train_idx, valid_idx) in enumerate(group_split(date_ids_model, NUM_SPLITS, GAP)):
        X_train, X_valid = X_model[train_idx], X_model[valid_idx]
        y_train, y_valid = y_model[train_idx], y_model[valid_idx]

        X_train = X_train.replace(np.inf, np.nan).replace(-np.inf, np.nan)
        X_valid = X_valid.replace(np.inf, np.nan).replace(-np.inf, np.nan)

        m = xgb.XGBRegressor(device="cuda", **xgb_params)
        m.fit(X_train, y_train, eval_set=[(X_valid, y_valid)], verbose=50, early_stopping_rounds=50)
        print(f"Fold {fold + 1} Training finished.")

        model_filename = f"model_xgb_fold_{fold + 1}.pkl"
        model_path = f"{OUTPUT_PATH}/model_xgb_fold_{fold + 1}.pkl"
        joblib.dump(m, model_path)
        file_names.append(model_filename)

        y_pred_valid = m.predict(X_valid)

        y_pred_valid = np.nan_to_num(y_pred_valid)
        y_valid = np.nan_to_num(y_valid)

        mae1 = mean_absolute_error(y_valid, y_pred_valid)

        mae_scores_kfold.append(mae1)
    average_mae_kfold = np.mean(mae_scores_kfold)
    print(f"{average_mae_kfold=}")


if __name__ == '__main__':
    train()
