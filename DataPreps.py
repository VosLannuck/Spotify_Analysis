import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sklearn

from sklearn.preprocessing import LabelEncoder
from omegaconf import OmegaConf, ListConfig, DictConfig
from typing import List, Dict, Tuple

from sklearn.preprocessing import StandardScaler


def normalizeDataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, StandardScaler]:
    scaler: StandardScaler = StandardScaler()
    scaler.fit(df)

    scaled_dataset: np.array = scaler.transform(df)
    scaled_df: pd.DataFrame = pd.DataFrame(scaled_dataset, columns=df.columns)
    return scaled_df, scaler


def concatDataset(df: pd.DataFrame, sec_df: pd.DataFrame,
                  num_cols: List[str]) -> pd.DataFrame:
    df = df.drop(num_cols, axis=1)
    return pd.concat([df, sec_df], axis=1)


def dropNa(df: pd.DataFrame) -> pd.DataFrame:
    df.dropna(inplace=True)
    return df


def encodeCategorical(df: pd.DataFrame,
                      cat_cols: List[str]) -> Tuple[List[LabelEncoder],
                                                    List[np.array]]:
    encoder_list: List[LabelEncoder] = []
    categorical_res_list: List[np.array] = []
    for col in cat_cols:
        encoder: LabelEncoder = LabelEncoder()
        encoded_res: np.array = encoder.fit_transform(df[col].values)

        categorical_res_list.append(encoded_res)
        encoder_list.append(encoder)
    return encoder_list, categorical_res_list


def run() -> Tuple[pd.DataFrame, pd.DataFrame]:
    config: Dict[DictConfig, ListConfig] = OmegaConf.load("./params.yaml")
    main_df: pd.DataFrame = pd.read_csv(config.data.datasetPath, index_col=[0])
    num_cols: List[str] = [
        config.constant.danceability_str,
        config.constant.loudness_str,
        config.constant.speechiness_str,
        config.constant.acousticness_str,
        config.constant.instrumentallness_str,
        config.constant.popularity_str,
        config.constant.liveness_str,
        config.constant.valence_str,
        config.constant.tempo_str,
        config.constant.duration_str
    ]

    cat_cols: List[str] = [
        config.constant.artists_str,
        config.constant.track_genre_str,
        config.constant.explicit_str
    ]

    drop_cols: List[str] = [
        config.constant.popularity_str,
        config.constant.track_id_str,
        config.constant.album_name_str,
        config.constant.track_name_str
    ]

    dropNa(main_df)
    num_df: pd.DataFrame = main_df.loc[:, num_cols]
    scaled_df, scaler = normalizeDataset(num_df)
    scaled_df.reset_index(drop=True, inplace=True)
    main_df.reset_index(drop=True, inplace=True)
    sec_main_df: pd.DataFrame = concatDataset(main_df, scaled_df, num_cols)

    cat_encoders, encoder_results = encodeCategorical(sec_main_df, cat_cols)
    for result, col in zip(encoder_results, cat_cols):
        sec_main_df[col] = result

    x: pd.DataFrame = sec_main_df.drop(drop_cols, axis=1)
    y: pd.DataFrame = sec_main_df[config.constant.popularity_str]

    return config, x, y


# if __name__ == "__main__":
#    x, y = run()
#    print(x.head(1))
#    print(x.shape)
#    print(y.shape)
