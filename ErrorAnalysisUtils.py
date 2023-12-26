import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt


def groupedAndSorted(df: pd.DataFrame,
                     group: str,
                     sort_by: str):
    grouped = df.groupby(group)
    df: pd.DataFrame = grouped[sort_by].mean()
    df = pd.DataFrame(df.sort_values(ascending=False))
    return df


def showPopularGenre(df:pd.DataFrame,
                     columns: str,
                     target: str,
                     n = 80):
    df_temp: pd.DataFrame = pd.DataFrame(columns=["genre", "percent"])

    for col in columns:
        df_t: pd.DataFrame = df[df['track_genre'] == col]
        df_t = df_t.sort_values(target, ascending=False )
        percentage = len(df_t[df_t[target] >= n]) / len(df_t) * 100
        new_obs = {'genre': col, 'percent': percentage}

        df_temp = pd.concat([df_temp, pd.DataFrame(new_obs,
                                                   index=[0])],
                            ignore_index=True)
        print(f"{col} genre has : {percentage} percent songs above {n} popularity")
    return df_temp.sort_values('percent', ascending=False)

