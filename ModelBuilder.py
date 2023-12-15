import pandas as pd
import numpy as np

import DataPreps
import mlflow

from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import mean_squared_error
from typing import List, Dict, Tuple
from omegaconf import DictConfig, ListConfig


def print_model_performance(y_train: np.array, y_train_pred,
                            y_val: np.array, y_val_pred,
                            model_name: str = "LinReg") -> Tuple[pd.DataFrame, pd.DataFrame]:
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

    train_residual_df["train_actual"] = y_train.values
    train_residual_df["train_pred"] = y_train_pred
    train_residual_df["train_residual"] = train_residual_df["train_pred"] - train_residual_df["train_actual"]

    val_residual_df["val_actual"] = y_val.values
    val_residual_df["val_pred"] = y_val_pred
    val_residual_df["val_residual"] = val_residual_df["val_pred"] - val_residual_df["val_actual"]

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


def decision_tree(
    config: Tuple[DictConfig, ListConfig],
    x_train: np.array, y_train: np.array,
    x_val: np.array, y_val: np.array
    ):
    dt_args = config.dt_args
    decisionTree: DecisionTreeRegressor = DecisionTreeRegressor(
        min_samples_leaf=dt_args.min_leaf,
        min_samples_split=dt_args.min_split,
    )
    decisionTree.fit(x_train, y_train)

    y_val_pred: np.array = decisionTree.predict(x_val)
    y_train_pred: np.array = decisionTree.predict(x_train)

    print_model_performance(y_train, y_train_pred,
                            y_val, y_val_pred, "DecisionTree"
                            )


def randomforest(config: Tuple[DictConfig, ListConfig],
                 x_train: np.array, y_train: np.array,
                 x_val: np.array, y_val: np.array
                 ):
    rf_args = config.rf_args
    randomForest: RandomForestRegressor = RandomForestRegressor(
        min_samples_leaf=rf_args.min_leaf,
        min_samples_split=rf_args.min_split,
    )
    randomForest.fit(x_train, y_train)

    y_val_pred: np.array = randomForest.predict(x_val)
    y_train_pred: np.array = randomForest.predict(x_train)

    print_model_performance(y_train, y_train_pred,
                            y_val, y_val_pred, "RandomForest"
                            )


def calculateMetrics(y_train: np.array, y_val: np.array,
                     y_train_pred: np.array,
                     y_val_pred: np.array) -> Tuple[float, float,
                                                    float, float]:
    mse_train: float = mean_squared_error(y_train, y_train_pred)
    mse_val: float = mean_squared_error(y_val, y_val_pred)

    rmse_train: float = mean_squared_error(y_train, y_train_pred) * 0.5
    rmse_val: float = mean_squared_error(y_val, y_val_pred) * 0.5
    print("RMSE Train: %g " % (rmse_train))
    print("RMSE VAL : %g" % (rmse_val))
    return mse_train, mse_val, rmse_train, rmse_val


def robust_fit_linreg(x: np.array, y: np.array, cv,
                      config: Tuple[DictConfig, ListConfig],
                      params: Dict = None,
                      modelName: str = "LinReg"
                      ) -> List[LinearRegression]:
    models: List[LinearRegression] = []
    with mlflow.start_run():
        mlflow.log_param("model_name", modelName)
        for i, (idx_train, idx_val) in enumerate(cv):
            x_train, y_train = x[idx_train], y[idx_train]
            x_val, y_val = x[idx_val], y[idx_val]
            model: LinearRegression = LinearRegression()
            model.fit(x_train, y_train)
            y_train_pred: np.array = model.predict(x_train)
            y_val_pred: np.array = model.predict(x_val)

            mse_train, mse_val, rmse_train, rmse_val = calculateMetrics(y_train,
                                                                        y_val,
                                                                        y_train_pred,
                                                                        y_val_pred)
            logMetric(config, mse_train,
                      mse_val, rmse_train,
                      rmse_val)

            models.append(model)

    return models


def logMetric(config, mse_train: float,
              mse_val: float, rmse_train: float,
              rmse_val: float):
    conf_ = config.mlflow
    mlflow.log_metric(conf_.mse_train, mse_train)
    mlflow.log_metric(conf_.mse_val, mse_val)
    mlflow.log_metric(conf_.rmse_train, rmse_train)
    mlflow.log_metric(conf_.rmse_val, rmse_val)

def 
def robust_fit_decisionTree():
    ...

def robust_fit_randomForest():
    ...
def robust_fit_knn():
    ...

def visualizeImportance(models: RegressorMixin, main_df_cols: List[str]):
    featureImportance: pd.DataFrame=pd.DataFrame(columns= main_df_cols)


def makeSplit(x: pd.DataFrame, y: pd.Series,
              n_splits: int = 5):
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    cv: List[Tuple[np.array, np.array]] = list(fold.split(x, y))
    return cv


def run_models():
    config, x, y = DataPreps.run()
    x, y = x.values, y.values
    ms_split = config.ms_split
    cv = makeSplit(x,y, n_splits=5)
    robust_fit_linreg(x, y, cv, config)

    #x_train, x_val, y_train, y_val = train_test_split(x, y,
    #                                                 test_size=ms_split.test_size,
    #                                                 random_state=ms_split.random_state)
    #randomforest(config, x_train, y_train, x_val, y_val)


if __name__ == "__main__":
    run_models()
