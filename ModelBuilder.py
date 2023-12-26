#%%
import pandas as pd
import numpy as np

import DataPreps
import torch
import mlflow
import seaborn as sns

import matplotlib.pyplot as plt

from torch.nn import (Module, Linear, ReLU)
from sklearn.model_selection import train_test_split, KFold
from sklearn.base import RegressorMixin
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.metrics import mean_squared_error
from typing import List, Dict, Tuple, Any, Union
from omegaconf import DictConfig, ListConfig

from Main_Enum import ModelName


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


def linreg(config: Union[DictConfig, ListConfig],
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


def knn(config: Union[DictConfig,
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
                config: Union[DictConfig, ListConfig],
                x_train: np.array, y_train: np.array,
                x_val: np.array, y_val: np.array):
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


def randomforest(config: Union[DictConfig, ListConfig],
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
    print("=========  ==========")
    print("RMSE Train: %g " % (rmse_train))
    print("RMSE VAL : %g" % (rmse_val))
    print("=========  ==========\n")
    return mse_train, mse_val, rmse_train, rmse_val


def getMisclassification(y_train, y_val,
                         y_train_pred, y_val_pred):
    miss_index_yTrain: np.array = np.where(y_train != y_train_pred)[0]
    miss_index_y_val: np.array = np.where(y_val != y_val_pred)[0]
    return miss_index_yTrain, miss_index_y_val


def errorAnalysis_fit(x_train: np.array, y_train: np.array,
                      x_val: np.array, y_val: np.array,
                      df: pd.DataFrame,
                      model: RegressorMixin,
                      modelName: str = "LinReg"):

    model.fit(x_train, y_train)
    y_train_pred = model.predict(x_train)
    y_val_pred = model.predict(x_val)
    calculateMetrics(y_train, y_val,
                     y_train_pred, y_val_pred)

    miss_indx_train, miss_indx_val = getMisclassification(y_train, y_val,
                                                          y_train_pred,
                                                          y_val_pred)
    df_miss = df.loc[miss_indx_train, :]
    df_miss['popularity'] = y_train[miss_indx_train]
    df_miss['predicted_train'] = y_train_pred[miss_indx_train]
    df_miss['model_name'] = modelName.name

    df_miss_val = df.loc[miss_indx_val, :]
    df_miss_val['popularity'] = y_val[miss_indx_val]
    df_miss_val['predicted_val'] = y_val_pred[miss_indx_val]
    df_miss_val['model_name'] = modelName.name

    return df_miss, df_miss_val


def preserveRealValues(df1: pd.DataFrame,
                       scaler, cat_encoder,
                       list_cat: List[str]
                       ):

    scaler_f: List[str] = scaler.feature_names_in_
    scaler_encoders: List = cat_encoder
    df1_copy: pd.DataFrame = df1.copy()
    df1_copy[scaler_f] = scaler.inverse_transform(df1[scaler_f].values)
    for i, scaler_cat in enumerate(scaler_encoders):
        df1_copy[list_cat[i]] = scaler_encoders[i].inverse_transform(df1[list_cat[i]].values)

    return df1_copy


def robust_fit(x: np.array, y: np.array, cv,
               config: Union[DictConfig, ListConfig],
               model: RegressorMixin,
               modelName: str = "LinReg"
               ) -> List[LinearRegression]:
    models: List[RegressorMixin] = []
    with mlflow.start_run():
        mlflow.log_param("model_name", modelName)
        for i, (idx_train, idx_val) in enumerate(cv):
            x_train, y_train = x[idx_train], y[idx_train]
            x_val, y_val = x[idx_val], y[idx_val]
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


def getImportance(models: List[Union[RegressorMixin,  Module]],
                  main_df_cols: List[str]):

    featureImportance: pd.DataFrame = pd.DataFrame()

    for indx, model in enumerate(models):
        _df = pd.DataFrame()
        _df['column'] = main_df_cols

        if (type(model) is LinearRegression):
            _df['feature_importance'] = model.coef_
        else:
            _df['feature_importance'] = model.feature_importances_
        _df['fold'] = indx + 1
        featureImportance = pd.concat([featureImportance, _df],
                                      axis=0,
                                      ignore_index=True)
    grouped = featureImportance.groupby('column')\
                              .sum()[['feature_importance']]\
                              .sort_values('feature_importance',
                                           ascending=False).index
    return featureImportance, grouped


def visualizeImportance(feature_df: pd.DataFrame,
                        grouped: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(8, max(6, len(grouped) * .25)))
    sns.boxenplot(data=feature_df,
                  x="feature_importance",
                  y="column",
                  order=grouped,
                  ax=ax,
                  orient="h"
                  )
    ax.tick_params(axis="x", rotation=0)
    ax.grid()
    fig.tight_layout()
    plt.show()


def makeSplit(x: pd.DataFrame, y: pd.Series,
              n_splits: int = 5) -> List[Tuple[np.array, np.array]]:
    fold = KFold(n_splits=n_splits, shuffle=True, random_state=1)
    cv: List[Tuple[np.array, np.array]] = list(fold.split(x, y))
    return cv


def preserve_params(config, modelName: ModelName):
    params: Dict[str, Any] = None
    if (modelName == ModelName.LIN_REG):
        params = {
            "n_jobs": config.linreg_args.n_jobs
        }
    elif (modelName == ModelName.DT):
        params = {
            "max_depth": config.dt_args.max_depth,
            "min_samples_split": config.dt_args.min_split,
            "min_samples_leaf": config.dt_args.min_leaf,
        }
    elif (modelName == ModelName.KNN):
        params = {
            "n_neighbors": config.knn_args.n_neighbors,
            "weights": config.knn_args.weights,
            "n_jobs": -1,
        }
    elif (modelName == ModelName.RF):
        params = {
            "n_estimators": config.rf_args.n_estimator,
            "min_samples_split": config.rf_args.min_split,
            "min_samples_leaf": config.rf_args.min_leaf,
            "max_depth": config.rf_args.max_depth
        }
    elif (modelName == ModelName.ADA_BST):
        params = {
            "n_estimators": config.ada_args.n_estimators,
            "learning_rate": config.ada_args.lr, "loss": config.ada_args.loss, "random_state": config.ada_args.random_state,
        }
    elif (modelName == ModelName.X_BST):
        ...
    elif (modelName == ModelName.MLP):
        ...
    return params


def craftLinearRegression(params: Dict = None):
    if params is None:
        params = {}
    return LinearRegression(**params)


def craftDecisionTree(params: Dict = None):
    if params is None:
        params = {}
    return DecisionTreeRegressor(**params)


def craftKNN(params: Dict = None):
    if params is None:
        params = {}
    return KNeighborsRegressor(**params)


def craftRandomForest(params: Dict = None):
    if params is None:
        params = {}
    return RandomForestRegressor(**params)


def craftLightGBM(params: Dict = None):
    if params is None:
        params = {}


def craftAdaBoost(params: Dict = None):
    if params is None:
        params = {}

    return AdaBoostRegressor(**params)


def craftMLP(params: Dict = None):
    if params is None:
        params = {}

    modelSequential = torch.nn.Sequential([
        Linear(in_features=params.in_features,
               out_features=params.default_features),
        ReLU(),
        Linear(in_features=params.default_features,
               out_features=params.default_features // 2),
        ReLU(),
        Linear(in_features=params.default_features // 2, out_features=1)
    ])

    return modelSequential


def preserveModel(modelName: ModelName,
                  config: Tuple[DictConfig, ListConfig]):
    model: RegressorMixin
    if (modelName == ModelName.DT):
        params = preserve_params(config, modelName)
        model = craftDecisionTree(params)
    elif (modelName == ModelName.LIN_REG):
        params = preserve_params(config, modelName)
        model = craftLinearRegression(params)
    elif (modelName == ModelName.RF):
        params = preserve_params(config, modelName)
        model = craftRandomForest(params)
    elif (modelName == ModelName.KNN):
        params = preserve_params(config, modelName)
        model = craftKNN(params)
    elif (modelName == ModelName.ADA_BST):
        params = preserve_params(config, modelName)
        model = craftAdaBoost(params)
    elif (modelName == ModelName.MLP):
        # Not yet finish
        params = preserve_params(config, modelName)
        model = craftMLP(params)

    return model

def run_model_error_analysis(config: Union[DictConfig, ListConfig],
                             modelName: ModelName, split: int,
                             target: str):
    x, y = DataPreps.run(config, target)
    predictors: List[str] = x.columns.values
    x, y = x.values, y.values
    model: Union[RegressorMixin, Module] = preserveModel(modelName,
                                                         config)
    

def run_models(config: Union[DictConfig, ListConfig],
               modelName: ModelName, split: int,
               target: str):
    x, y, _, _ = DataPreps.run(config, target)
    predictors: List[str] = x.columns.values
    x, y = x.values, y.values
    cv = makeSplit(x, y, n_splits=split)
    model: Union[RegressorMixin, Module] = preserveModel(modelName,
                                                         config)
    models = robust_fit(x, y, cv, config, model)
    feature_df, grouped = getImportance(models, predictors)
    visualizeImportance(feature_df, grouped)

    # Only if you want to test simple model and dont want to run robust_fit
    # Used for Analysis Error

    # ms_split = config.ms_split
    # x_train, x_val, y_train, y_val = train_test_split(x, y,
    #                                                 test_size=ms_split.test_size,
    #                                                 random_state=ms_split.random_state)
    # randomforest(config, x_train, y_train, x_val, y_val)


"""
if __name__ == "__main__":
    run_models()
"""
