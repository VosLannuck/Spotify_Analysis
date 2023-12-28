import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List


def groupedAndSorted(df: pd.DataFrame,
                     group: str,
                     sort_by: str):
    grouped = df.groupby(group)
    df: pd.DataFrame = grouped[sort_by].mean()
    df = pd.DataFrame(df.sort_values(ascending=False))
    return df


def getPercentageBasedCol(df: pd.DataFrame,
                          cat_column: str,
                          columns: str,
                          target: str,
                          verbose: bool = False,
                          n=80):
    df_temp: pd.DataFrame = pd.DataFrame(columns=[cat_column, "percent"])

    for col in columns:
        df_t: pd.DataFrame = df[df[cat_column] == col]
        df_t = df_t.sort_values(target, ascending=False )
        percentage = len(df_t[df_t[target] >= n]) / len(df_t) * 100
        new_obs = {cat_column: col, 'percent': percentage}

        df_temp = pd.concat([df_temp, pd.DataFrame(new_obs,
                                                   index=[0])],
                            ignore_index=True)
        if verbose == True:
            print(f"{col} {cat_column} has : {percentage} percent songs above {n} {target}")

    return df_temp.sort_values('percent', ascending=False)


def makePie(data: np.ndarray, labels: np.ndarray, title: str):
    ax = plt.pie(x=data, labels=labels)
    #plt.title(title)
    plt.show()


def makeHorizontalBar(x: np.ndarray, y: np.ndarray,
                      title: str="Residual percentage", ylim=None):
    sns.barplot(x=x, y=y, hue=x, )
    sns.despine(left=True, bottom=True)
    if ylim:
        plt.ylim(ylim)
    plt.tick_params(axis='x', rotation=90)
    plt.title(title)
    plt.show()


def getNSongsBasedOnGenre(df: pd.DataFrame,
                         genre: str,
                         n: int,
                         sort_by: str= "popularity") -> pd.DataFrame:
    genreSeries: pd.Series = df.loc[df['track_genre'] == genre]
    if (len(genreSeries) <= 0):
        raise ("Damn bro no genre")
    return genreSeries.sort_values(sort_by, ascending=False)[:n]


def plotResidualOfColumns(true_values: np.ndarray,
                             pred_values: np.ndarray):
    residual: np.ndarray = pred_values - true_values
    sns.histplot( x=residual, kde=True, bins=100)
    plt.show()


def filterResidual(df: pd.DataFrame, col_true: str,
                   col_pred: str,
                   n1: int = 0, n2: int = 10):

    true_val: np.ndarray = df[col_true].values
    pred_val: np.ndarray = df[col_pred].values
    indx_filtered: np.ndarray = filterBasedResidual(true_val, pred_val, n1, n2)
    return df.iloc[indx_filtered, :]


def filterBasedResidual(true_values: np.ndarray,
                        pred_values: np.ndarray,
                        n1: int, n2: int):
    residual: np.ndarray = np.abs(pred_values - true_values)
    filtered: np.ndarray = np.where((residual >= n1) & (residual <= n2))
    return filtered[0]


def plotHistOfTheTarget(true: np.ndarray, pred: np.ndarray):

    sns.histplot(true, label="Train True", color='C1', bins=30)
    sns.histplot(pred, label="Train Pred", color="black", bins=30)
    plt.show()


def plotErrorAnalysisCat_num(df_real: pd.DataFrame, cat_column: str,
                      target_column: str, pred_colum: str,
                      limit_resid_1: int, limit_resid_2: int,
                      threshold: int, n_pie: int):
    real_array: np.ndarray = df_real[target_column].values
    pred_array: np.array = df_real[pred_colum].values
    filtered_res: np.ndarray = filterResidual(df_real, target_column,
                                              pred_colum,  limit_resid_1, 
                                              limit_resid_2)
    filtered_rate: pd.DataFrame = groupedAndSorted(filtered_res, cat_column, target_column )

    percentage_df = getPercentageBasedCol(filtered_res, cat_column, filtered_rate.index,
                                          target_column, threshold)[:n_pie]
    print(percentage_df)
    plotResidualOfColumns(real_array,
                          pred_array)
    #makePie(percentage_df['percent'], labels=percentage_df[cat_column],
     #       title="Not")
    
    makeHorizontalBar(percentage_df['percent'], percentage_df[cat_column],)

def plotHistOfPrediction(df_real_1, df_real_2,
                         target_column: str="popularity",
                         pred_column_train: str="predicted_train",
                         pred_column_val: str="predicted_val"):

    real_array: np.ndarray = df_real_1[target_column].values
    pred_array: np.array = df_real_1[pred_column_train].values
    val_array: np.ndarray = df_real_2[target_column].values
    val_pred: np.array = df_real_2[pred_column_val].values
    plotHistOfTheTarget(real_array, pred_array)
    plotHistOfTheTarget(val_array, val_pred)


