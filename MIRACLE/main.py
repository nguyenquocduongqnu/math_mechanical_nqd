import argparse
import os
import random
import sys
from typing import Any
import torch
import numpy as np
import pandas as pd
from signal import SIGINT, signal
from sklearn.preprocessing import StandardScaler
import tensorflow.compat.v1 as tf
import networkx as nx

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__ ),'miracle')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__ ),'tests')))

# miracle absolute
from miracle.MIRACLE import MIRACLE
from miracle.compare_methods import load_imputer

tf.disable_v2_behavior()
np.set_printoptions(suppress=True)


def handler(signal_received: Any, frame: Any) -> None:
    print("SIGINT or CTRL-C detected. Exiting gracefully")
    exit(0)

def binary_sampler(p: float, rows: int, cols: int) -> np.ndarray:
    """Sample binary random variables.

    Args:
      - p: probability of 1
      - rows: the number of rows
      - cols: the number of columns

    Returns:
      - binary_random_matrix: generated binary random matrix.
    """
    unif_random_matrix = np.random.uniform(0.0, 1.0, size=[rows, cols])
    binary_random_matrix = 1 * (unif_random_matrix < p)
    return binary_random_matrix

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--reg_lambda", type=float, default=0.1)
    parser.add_argument("--reg_beta", type=float, default=0.1)
    parser.add_argument("--gpu", type=str, default="")
    parser.add_argument("--ckpt_file", type=str, default="tmp.ckpt")
    parser.add_argument("--window", type=int, default=10)
    parser.add_argument("--max_steps", type=int, default=400)
    parser.add_argument("--missingness", type=float, default=0.2)
    args = parser.parse_args()
    signal(SIGINT, handler)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    data = pd.read_csv(r'test_dataset_prob_dl_no_null.csv').drop(['Unnamed: 0'], axis=1).drop(['Output'], axis=1)
    df = data.sample(frac=1).reset_index(drop=True)
    print(df.head())
    #
    # scaler = StandardScaler()
    # df = scaler.fit_transform(df)

    X_MISSING = df.copy()  # This will have missing values
    X_TRUTH = df.copy()  # This will have no missing values for testing

    # Generate MCAR
    X_MASK = binary_sampler(1 - args.missingness, X_MISSING.shape[0], X_MISSING.shape[1])
    X_MISSING[X_MASK == 0] = np.nan

    datasize = X_MISSING.shape[0]
    missingness = args.missingness
    feature_dims = X_MISSING.shape[1]

    print(
        f"""
        Datasize = {datasize}
        Missingness = {missingness}
        NumFeats =  {feature_dims}
        """
    )

    # Append indicator variables - One indicator per feature with missing values.
    missing_idxs = np.where(np.any(np.isnan(X_MISSING), axis=0))[0]

    seed_imputer_missforest = load_imputer("missforest")
    X_seed_imputer_missforest = seed_imputer_missforest.fit_transform(X_MISSING)

    seed_imputer_mean = load_imputer("mean")
    X_seed_imputer_mean = seed_imputer_mean.fit_transform(X_MISSING)

    seed_imputer_mice = load_imputer("mice")
    X_seed_imputer_mice = seed_imputer_mice.fit_transform(X_MISSING)

    seed_imputer_gain = load_imputer("gain")
    X_seed_imputer_gain = seed_imputer_gain.fit_transform(X_MISSING)

    seed_imputer_knn = load_imputer("knn")
    X_seed_imputer_knn = seed_imputer_knn.fit_transform(X_MISSING)

    # Initialize MIRACLE
    miracle = MIRACLE(
        num_inputs=X_MISSING.shape[1],
        reg_lambda=args.reg_lambda,
        reg_beta=args.reg_beta,
        n_hidden=32,
        ckpt_file=args.ckpt_file,
        missing_list=missing_idxs,
        reg_m=0.1,
        lr=0.0001,
        window=args.window,
        max_steps=args.max_steps,
    )

    # Train MIRACLE
    miracle_imputed_data_x = miracle.fit(
        X_MISSING,
        X_seed=X_seed_imputer_mean,
    )

    print(f"RMSE of MEAN: {miracle.rmse_loss(X_TRUTH, X_seed_imputer_mean, X_MASK)}")
    print(f"RMSE of KNN: {miracle.rmse_loss(X_TRUTH, X_seed_imputer_knn, X_MASK)}")
    print(f"RMSE of MICE: {miracle.rmse_loss(X_TRUTH, X_seed_imputer_mice, X_MASK)}")
    print(f"RMSE of GAIN: {miracle.rmse_loss(X_TRUTH, X_seed_imputer_gain, X_MASK)}")
    print(f"RMSE of MISSFOREST: {miracle.rmse_loss(X_TRUTH, X_seed_imputer_missforest, X_MASK)}")
    print(f"RMSE of MIRACLE:  {miracle.rmse_loss(X_TRUTH, miracle_imputed_data_x, X_MASK)}")
    print(X_seed_imputer_missforest)

