import pandas as pd
import numpy as np

import DataPreps

from sklearn.model_selection import train_test_split, KFold
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from typing import List, Dict, Tuple
from omegaconf import DictConfig, ListConfig


def print_model_performance(y_train: np.array, y_train_pred,
                            y_val: np.array, y_val_pred,
                            model_name: str = "LinReg") -> Tuple[pd.DataFrame,
                                                                 pd.DataFrame]:
    print("Benchmark of %s model" % (model_name))
    print("MSE - Train: %g" % (mean_squared_error(y_train, y_train_pred)))
    print("MSE - Validation: %g " % (mean_squared_error(y_val, y_val_pred)))
    print("\n")
    print("RMSE - Train: %g" % (np.sqrt(mean_squared_error(y_train,
                                                           y_train_pred))))

    print("RMSE - Validation: %g " % (np.sqrt(mean_squared_error(y_val,
                                                                 y_val_pred))))
    train_residual_df: pd.DataFrame = pd.DataFrame(columns=["train_actual",
                                                            "train_pred",
                                                            "train_residual"])

    val_residual_df: pd.DataFrame = pd.DataFrame(columns=["val_actual",
                                                          "val_pred",
                                                          "val_residual"])
    train_residual_df["train_actual"] = y_train
    train_residual_df["train_pred"] = y_train_pred
    train_residual_df["train_residual"] = train_residual_df["train_pred"] - ["train_actual"]

    val_residual_df["val_actual"] = y_val
    val_residual_df["val_pred"] = y_val_pred
    val_residual_df["val_residual"] = val_residual_df["val_pred"] - ["val_actual"]

    return train_residual_df, val_residual_df


def linreg(config: Tuple[DictConfig, ListConfig],
           x_train: np.array, y_train: np.array,
           x_val: np.array, y_val: np.array
           ):
    linReg: LinearRegression = LinearRegression()
    linReg.fit(x_train, y_train)

    y_val_pred: np.array = linReg.predict(x_val)
    y_train_pred: np.array = linReg.predict(x_train)

    print_model_performance(y_train, y_train_pred,
                            y_val, y_val_pred, "LinReg"
                            )


def knn(config: Tuple[DictConfig,
        ListConfig],
        x_train: np.array, y_train: np.array,
        x_val: np.array, y_val: np.array):
    knn_args = config.knn_args

    knn: KNeighborsRegressor = KNeighborsRegressor(
                                                   n_neighbors=knn_args.n_neighbors,
                                                   weights=knn_args.weights,
                                                   p=knn_args.p
                                                   )

    knn.fit(x_train, y_train)

    y_val_pred: np.array = knn.predict(x_val)
    y_train_pred: np.array = knn.predict(x_train)

    print_model_performance(y_train, y_train_pred,
                            y_val, y_val_pred, "KNN"
                            )

def decision_tree(config: Tuple[DictConfig, ListConfig],
           x_train: np.array, y_train: np.array,
           x_val: np.array, y_val: np.array
           ):

def run_models():
    config, x, y = DataPreps.run()
    x_train, x_val, y_train, y_val = train_test_split(x, y,
                                                      test_size=config.ms_split.test_size,
                                                      random_state=config.ms_split.random_state)
    knn(config, x_train, y_train, x_val, y_val)



if __name__ == "__main__":
    run_models()
