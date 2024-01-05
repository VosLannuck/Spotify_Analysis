#%%
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pprint
import warnings
import statsmodels.api as sm

from scipy import stats
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,silhouette_score
from sklearn.cluster import KMeans, DBSCAN, OPTICS
from mpl_toolkits.mplot3d import Axes3D
import plotly.graph_objects as go

from sklearn.model_selection import train_test_split
from typing import List, Dict, Any
warnings.filterwarnings("ignore")
SEED : int = 5
np.random.seed(SEED)
#%%
"""
    Uni Variate & Getting know the Dataset
"""
FILENAME : str = "dataset.csv"
df : pd.DataFrame = pd.read_csv(FILENAME)

def PrintColumnsDescription():
    
    pprint.pprint(""" 
track_id: The Spotify ID for the track
artists: The artists' names who performed the track.
album_name: The album name in which the track appears
track_name: Name of the track
popularity: The popularity of a track is a value between 0 and 100, with 100 being the most popular.
duration_ms: The track length in milliseconds
explicit: Whether or not the track has explicit lyrics (true = yes it does; false = no it does not OR unknown)
danceability: Describes how suitable a track is for dancing A value between 0.0 and 1.0.
energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.
key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1
loudness: The overall loudness of a track in decibels (dB)
mode: Indicates the modality (major or minor) of a track. Major is represented by 1 and minor is 0
speechiness: Speechiness detects the presence of spoken words in a track. Value Between 0 and 1.
acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic
instrumentalness: Predicts whether a track contains no vocals. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content
liveness: Detects the presence of an audience in the recording. A value between 0.0 and 1.0.
valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.
tempo: The overall estimated tempo of a track in beats per minute (BPM).
time_signature: A notational convention to specify how many beats are in each bar (or measure).
track_genre: The genre in which the track belongs""")


def PrintColumnTypes():
    print(df.dtypes)
    
def PrintColumnsNull():
    print(df.isna().sum())

def PrintDuplicatedInfo():
    duplicated : pd.DataFrame = df[df["track_id"].duplicated("last")]
    print(duplicated.shape)
    
    duplicatedBasedTrack : pd.Series = duplicated.groupby("track_id")["track_id"].count()
    print(f"Total Duplicated observations : {duplicated.shape[0]}")
    print(f"Total Non-Duplicated observations : {df.shape[0] - duplicated.shape[0]}")

def PrintMoreInfoAboutArtist():
    n : int = 5
    print("Artists Column Information")
    print(f"Total Nunique : {df['artists'].nunique()}")
    
    f, ax= plt.subplots(nrows=1, ncols=2, figsize=(20,6),
                        constrained_layout=True)
    f.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    x : np.ndarray = df['artists'].value_counts().index
    y : np.ndarray = df["artists"].value_counts().values
    sns.barplot(x=x[0:n], y=y[0:n], ax= ax[0])
    ax[0].set_title(f"Top {n} Most Contributed Artist")
    
    sns.barplot(x=x[-n:], y=y[-n:], ax=ax[1])
    ax[1].set_title(f"Top {n} Least Contributed Artist")
    ax[1].set_ylim(0,20)
    plt.show()

def PrintMoreInfoAboutAlbum():
    n : int = 5
    columnName : str = "album_name"
    print("Album Column Information")
    print(f"Total Nunique : {df[columnName].nunique()}")
    
    f, ax= plt.subplots(nrows=1, ncols=2, figsize=(20,6),
                        constrained_layout=True)
    f.subplots_adjust(left=0.1, right=0.9, top=0.9, bottom=0.1)
    
    x : np.ndarray = df[columnName].value_counts().index
    y : np.ndarray = df[columnName].value_counts().values
    sns.barplot(x=x[0:n], y=y[0:n], ax= ax[0])
    ax[0].set_title(f"Top {n} total songs in Album")
    ax[0].set_xticklabels(x[0:n], rotation=90)
    
    sns.barplot(x=x[-n:], y=y[-n:], ax=ax[1])
    ax[1].set_title(f"Top Least {n} total songs in Album")
    ax[1].set_xticklabels(x[-n:], rotation=90)
    ax[1].set_ylim(0,20)
    plt.show()

def PrintMoreInfoAboutPopularity():
    """ popularity: The popularity of a track is a value between 0 and 100, with 100 being the most popular.
    """
    columnName : str = 'popularity'
    plt.figure(figsize=( 5, 5))
    sns.kdeplot(df[columnName], fill=True)
    plt.title("Popularity KDE Plot")
    plt.show()


def PrintMoreAboutDuration():
    columnName : str = "duration_ms"
    secondsDur : pd.DataFrame = df[columnName] / 1000
    
    plt.figure(figsize=(5, 3), dpi=100)
    sns.kdeplot(secondsDur, fill=True)
    plt.title("Songs Duration")
    plt.xlabel("Track Duration in Seconds ")
    plt.show()
    

def PrintMoreAboutExplicit():
    """ 
    explicit: Whether or not the track has explicit lyrics (true = yes it does; false = no it does not OR unknown)
    """
    
    columnName: str = "explicit"

    f, ax = plt.subplots(nrows=1, ncols=1 ,figsize=(8 , 10))
   
    x : np.ndarray = df[columnName].value_counts().index
    y : np.ndarray = df[columnName].value_counts().values
   
    sns.barplot(x=x, y=y,ax=ax)
    ax.set_title("Total Explicit Songs")
    ax.set_ylabel("Counts")
    plt.plot()
    
def PrintMoreAboutDanceabillity():
    """ 
    danceability: Describes how suitable a track is for dancing A value between 0.0 and 1.0.
    """
    
    columnName : str = "danceability"
    plt.figure(figsize=(3,3), dpi=100)
    sns.kdeplot(df[columnName], fill=True)
    
    sns.lineplot(x = (0.5 for i in range(0, 3)), y=range(0,3), color='Red')   
    plt.xlabel("Danceability")
    plt.show()

def PrintMoreAboutEnergy():
    """ 
    energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.
    """
    
    columnName : str = "energy"
    plt.figure(figsize=(3,3), dpi=100)
    sns.kdeplot(df[columnName], fill=True)
    
    sns.lineplot(x = (0.5 for i in range(0, 3)), y=range(0,3), color='Red')   
    plt.xlabel("Energy")
    plt.show()

def PrintMoreAboutKey():
    """ 
    key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1
    """
    columnName : str = "key"
    plt.figure(figsize=(10, 3), dpi=100)
    x : np.ndarray = df[columnName].value_counts().index
    y : np.ndarray = df[columnName].value_counts().values
    sns.barplot(x=x, y=y)
    plt.xlabel("Key")
    plt.ylabel("Total Tracks")
    plt.show()

def PrintMoreAboutLoudness():
    """ 
    loudness: The overall loudness of a track in decibels (dB)
    """
    
    columnName : str = "loudness"
    plt.figure(figsize=(5, 3), dpi=100)
    sns.kdeplot(df[columnName], fill=True)
    plt.xlabel("loudness")
    plt.show()
    
def PrintMoreAboutMode():
    """ 
    mode: Indicates the modality (major or minor) of a track. Major is represented by 1 and minor is 0
    """
    columnName : str = "mode"
    plt.figure(figsize=(5, 3), dpi=100)
    x : np.ndarray = df[columnName].value_counts().index
    y : np.ndarray = df[columnName].value_counts().values
    sns.barplot(x=x, y=y)
    plt.xlabel("Mode (1 = Major, 0 = Minor)")
    plt.ylabel("Total Tracks")
    plt.show()

def PrintMoreAboutSpeechiness():
    """ 
    speechiness: Speechiness detects the presence of spoken words in a track. Value Between 0 and 1.
    """

    columnName : str = "speechiness"
    sns.kdeplot(df[columnName], fill=True)
    plt.xlabel(columnName)
    plt.show()

def PrintMoreAboutAcousticness():
    """ 
    acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic
    """
    plt.figure(figsize=(5,3))
    columnName : str ="acousticness"
    sns.kdeplot(df[columnName], fill=True)
    plt.show()

def PrintMoreAboutInstrumentalness():
    """ 
    instrumentalness: Predicts whether a track contains no vocals. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content
    """
    
    plt.figure(figsize=(5,3))
    columnName : str = "instrumentalness"
    sns.kdeplot(df[columnName], fill=True)
    plt.show()

def PrintMoreAboutLiveness():
    """ 
    liveness: Detects the presence of an audience in the recording. A value between 0.0 and 1.0.
    """
    plt.figure(figsize=(5,3))
    columnName : str = "liveness"
    sns.kdeplot(df[columnName], fill=True)
    plt.show()

def PrintMoreAboutTempo():
    """ 
    tempo: The overall estimated tempo of a track in beats per minute (BPM).
    """
    plt.figure(figsize=(5,3))
    columnName : str = "tempo"
    sns.kdeplot(df[columnName], fill=True)
    plt.show()

def PrintMoreAboutValence():
    """ 
    valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.
    """
    plt.figure(figsize=(5,3))
    columnName : str = "valence"
    sns.kdeplot(df[columnName], fill=True)
    plt.show()


def PrintMoreAboutTrackGenre():
    """ 
    track_genre: The genre in which the track belongs
    """
    columnName : str = "track_genre"
    n : int = 20
    f, ax = plt.subplots(nrows=2, ncols=1)
    
    x : np.ndarray = df[columnName].value_counts().index
    y : np.ndarray = df[columnName].value_counts().values
    sns.barplot(x = x[:n], y= y[:n],ax=ax[0])
    ax[0].set(title=f"Head : {n} Genres / {len(x)}")
    ax[0].set_xticklabels(x[:n], rotation=90)
    
    sns.barplot(x = x[-n:], y=y[-n:], ax=ax[1])
    ax[1].set(title=f"Tail : {n} Genres / {len(x)}")
    plt.subplots_adjust(wspace=0.6, hspace=1.5)
    ax[1].set_xticklabels(x[-n:], rotation=90)
    plt.show()


# BIVARIATE ANALYSIS 
ARTISTS : str = "artists"
TRACK_GENRE : str = "track_genre"
ALBUM_NAME : str = "album_name"
EXPLICIT : str = "explicit"
TRACK_NAME : str ="track_name"

## NUMERICAL_COL
POPULARITY : str = "popularity"
LOUDNESS : str ="loudness"
DANCEABILITY : str = "danceability"
LIVENESS : str = "liveness"

SPEECHINESS : str ="speechiness"
ACOUSTICNESS : str = "acousticness"
INSTRUMENTALNESS : str = "instrumentalness"
VALENCE : str = "VALENCE"
POPULARITY_COMPARABLE_VALUES : List[str] = [
    ARTISTS, TRACK_GENRE, ALBUM_NAME, EXPLICIT, TRACK_NAME
]

LOUDNESS_COMPARABLE_VALUES : List[str] = [
    ARTISTS, TRACK_GENRE, ALBUM_NAME, EXPLICIT
]

DANCEABILITY_COMPARABLE_VALUES : List[str] = [
    ARTISTS, TRACK_GENRE, ALBUM_NAME, EXPLICIT
]

LIVENESS_COMPARABLE_VALUES : List[str]  = [
    ARTISTS, ALBUM_NAME, EXPLICIT, TRACK_GENRE    
]

TOP_POPULARITY : int = 5
# [ Comparable Value : artists, track_genre, album_name, explicit, track_name]
# ( Constant : Popularity )
def Bi_PrintInfoPopularity(
    col_popularity : str , col_another : str,
    title_plot_1 : str,
    xlabel : str, ylabel : str, n : int = 5,
    nrows : int = 1, ncols : int = 1):
    
    AlbumSumPopularity: pd.DataFrame = df.groupby(col_another)[col_popularity].sum()
    sortedValues : pd.DataFrame = AlbumSumPopularity.sort_values(ascending=False)
    x : np.ndarray = sortedValues.index
    y : np.ndarray = sortedValues.values

    f, ax = plt.subplots(nrows=nrows, ncols=ncols)
    sns.barplot(x = x[:n], y=y[:n], ax=ax)
    
    ax.set_xticklabels(x[:n], rotation=90)
    ax.set_title(title_plot_1 if len(x) > 4 else f"By {col_another}")
    ax.set_ylabel(ylabel)
    
    plt.show()

# [ Comparable Value : artist, track_genre, album_name, explicit ]
# ( Constant : loudness )
def Bi_PrintInfoLoudness(
    col_loudness : str, col_another : str,
    title_plot_1 : str , title_plot_2 : str,
    xlabel : str, ylabel : str ,
    n : int = 5,
    nrows : int = 1, ncols : int = 2
):
  
  loudnessSum : pd.DataFrame = df.groupby(col_another)[col_loudness].aggregate("mean")
  
  sortedValues : pd.DataFrame = loudnessSum.sort_values(ascending=False)
  
  x : np.ndarray = sortedValues.index
  y : np.ndarray = sortedValues.values
  
  f, ax = plt.subplots(nrows=nrows, ncols=ncols)
  
  sns.barplot(x=x[:n], y=y[:n], ax=ax[0])
  sns.barplot(x=x[-n:], y=y[-n:], ax=ax[1])
  
  ax[0].set_xticklabels(x[:n],rotation=90)
  ax[0].set_title(title_plot_1)
  
  ax[1].set_xticklabels(x[-n:],rotation=90)
  ax[1].set_title(title_plot_2)
  
  plt.subplots_adjust(wspace=0.6, hspace=2.5)
  plt.show()

def Bi_PrintInfoDanceability(
    col_danceability : str, col_another : str,
    title_plot_1 : str, title_plot_2 : str,
    xlabel : str, ylabel : str, 
    n : int = 5,
    nrows : int = 1, ncols : int = 1
):
    danceAbillityGroup : pd.DataFrame = df.groupby(col_another)[col_danceability].aggregate("mean")
    sortedValuesBest : pd.DataFrame = danceAbillityGroup.sort_values(ascending=False)
    sortedValuesMedToHigh : pd.DataFrame = danceAbillityGroup[(danceAbillityGroup.values >= 0.4) & (danceAbillityGroup.values <= 0.8)].sort_values(ascending=False)
    sortedValuesLow : pd.DataFrame = danceAbillityGroup[(danceAbillityGroup.values >= 0.0) & (danceAbillityGroup.values < 0.4)].sort_values(ascending=False)
    
    x : np.ndarray = sortedValuesBest.index
    y : np.ndarray = sortedValuesBest.values
    
    x_1 : np.ndarray = sortedValuesMedToHigh.index
    y_1 : np.ndarray = sortedValuesMedToHigh.values
    
    x_2 : np.ndarray = sortedValuesLow.index
    y_2 : np.ndarray = sortedValuesLow.values
    
    f, ax = plt.subplots(nrows=nrows, ncols=ncols)
    
    xPlot : np.array = np.array(["Best", "Medium", "Low"])
    yPlot : np.array = np.array([y.sum(), y_1.sum(), y_2.sum()])
    
    sns.barplot(x=xPlot, y=yPlot,ax=ax)
    ax.set_ylabel("Total")
    
    plt.show()

def Bi_PrintInfoLiveness(
    col_liveness : str, col_another : str,
    title_plot_1 : str, title_plot_2 : str,
    xlabel : str, ylabel : str, 
    n : int = 5,
    nrows : int = 1, ncols : int = 1
):
    
    livenessGroup : pd.DataFrame = df.groupby(col_another)[col_liveness].aggregate("mean")
    sortedValuesLow : pd.DataFrame = livenessGroup[(livenessGroup.values > 0.8)].sort_values(ascending=False)
    sortedValuesMedToHigh : pd.DataFrame = livenessGroup[(livenessGroup.values >= 0.6) & (livenessGroup.values <= 0.8)].sort_values(ascending=False)
    sortedValuesLive : pd.DataFrame = livenessGroup[(livenessGroup.values >= 0.0) & (livenessGroup.values < 0.6)].sort_values(ascending=False)
    
    x : np.ndarray = sortedValuesLive.index
    y : np.ndarray = sortedValuesLive.values
    
    x_1 : np.ndarray = sortedValuesMedToHigh.index
    y_1 : np.ndarray = sortedValuesMedToHigh.values
    
    x_2 : np.ndarray = sortedValuesLow.index
    y_2 : np.ndarray = sortedValuesLow.values
    
    f, ax = plt.subplots(nrows=nrows, ncols=ncols)
    
    xPlot : np.array = np.array(["Live", "Medium", "Low"])
    yPlot : np.array = np.array([y.sum(), y_1.sum(), y_2.sum()])
    
    sns.barplot(x=xPlot, y=yPlot,ax=ax)
    ax.set_ylabel("Total")
    
    plt.show()

def Bi_PrintInfoSpeechiness(
    col_liveness : str, col_another : str,
    title_plot_1 : str, title_plot_2 : str,
    xlabel : str, ylabel : str, 
    n : int = 5,
    nrows : int = 1, ncols : int = 1
):
    
    speechinessGroup : pd.DataFrame = df.groupby(col_another)[col_liveness].aggregate("mean")
    sortedValuesHigh : pd.DataFrame = speechinessGroup [(speechinessGroup.values > 0.6)].sort_values(ascending=False)
    sortedValuesMedToHigh : pd.DataFrame = speechinessGroup[(speechinessGroup.values > 0.2) & (speechinessGroup.values <= 0.6)].sort_values(ascending=False)
    sortedValuesLow : pd.DataFrame = speechinessGroup[(speechinessGroup.values >= 0.0) & (speechinessGroup.values <= 0.2)].sort_values(ascending=False)
    
    x : np.ndarray = sortedValuesHigh.index
    y : np.ndarray = sortedValuesHigh.values
    
    x_1 : np.ndarray = sortedValuesMedToHigh.index
    y_1 : np.ndarray = sortedValuesMedToHigh.values
    
    x_2 : np.ndarray = sortedValuesLow.index
    y_2 : np.ndarray = sortedValuesLow.values
    
    f, ax = plt.subplots(nrows=nrows, ncols=ncols)
    
    xPlot : np.array = np.array(["High", "Medium", "Low"])
    yPlot : np.array = np.array([y.sum(), y_1.sum(), y_2.sum()])
    
    sns.barplot(x=xPlot, y=yPlot,ax=ax)
    ax.set_ylabel("Total")
    
    plt.show()


def Bi_PrintInfoInstrumentalness(
    col_instrumentalness : str, col_another : str,
    title_plot_1 : str, title_plot_2 : str,
    xlabel : str, ylabel : str, 
    n : int = 5,
    nrows : int = 1, ncols : int = 1
):
    
    instrumentalnessGroup : pd.DataFrame = df.groupby(col_another)[col_instrumentalness].aggregate("mean")
    sortedValuesHigh : pd.DataFrame = instrumentalnessGroup [(instrumentalnessGroup.values >= 0.8)].sort_values(ascending=False)
    sortedValuesMedToHigh : pd.DataFrame = instrumentalnessGroup[(instrumentalnessGroup.values > 0.2) & (instrumentalnessGroup.values < 0.8)].sort_values(ascending=False)
    sortedValuesLow : pd.DataFrame = instrumentalnessGroup[(instrumentalnessGroup.values >= 0.0) & (instrumentalnessGroup.values <= 0.2)].sort_values(ascending=False)
    
    x : np.ndarray = sortedValuesHigh.index
    y : np.ndarray = sortedValuesHigh.values
    
    x_1 : np.ndarray = sortedValuesMedToHigh.index
    y_1 : np.ndarray = sortedValuesMedToHigh.values
    
    x_2 : np.ndarray = sortedValuesLow.index
    y_2 : np.ndarray = sortedValuesLow.values
    
    f, ax = plt.subplots(nrows=nrows, ncols=ncols)
    
    xPlot : np.array = np.array(["High", "Medium", "Low"])
    yPlot : np.array = np.array([y.sum(), y_1.sum(), y_2.sum()])
    
    sns.barplot(x=xPlot, y=yPlot,ax=ax)
    ax.set_ylabel("Total")
    
    plt.show()


def Bi_PrintInfoAcousticness(
    col_acousticness : str, col_another : str,
    title_plot_1 : str, title_plot_2 : str,
    xlabel : str, ylabel : str, 
    n : int = 5,
    nrows : int = 1, ncols : int = 1
):
    
    acousticnessGroup : pd.DataFrame = df.groupby(col_another)[col_acousticness].aggregate("mean")
    sortedValuesHigh : pd.DataFrame = acousticnessGroup [(acousticnessGroup.values >= 0.8)].sort_values(ascending=False)
    sortedValuesMedToHigh : pd.DataFrame = acousticnessGroup[(acousticnessGroup.values > 0.2) & (acousticnessGroup.values < 0.8)].sort_values(ascending=False)
    sortedValuesLow : pd.DataFrame = acousticnessGroup[(acousticnessGroup.values >= 0.0) & (acousticnessGroup.values <= 0.2)].sort_values(ascending=False)
    
    x : np.ndarray = sortedValuesHigh.index
    y : np.ndarray = sortedValuesHigh.values
    
    x_1 : np.ndarray = sortedValuesMedToHigh.index
    y_1 : np.ndarray = sortedValuesMedToHigh.values
    
    x_2 : np.ndarray = sortedValuesLow.index
    y_2 : np.ndarray = sortedValuesLow.values
    
    f, ax = plt.subplots(nrows=nrows, ncols=ncols)
    
    xPlot : np.array = np.array(["High", "Medium", "Low"])
    yPlot : np.array = np.array([y.sum(), y_1.sum(), y_2.sum()])
    
    sns.barplot(x=xPlot, y=yPlot,ax=ax)
    ax.set_ylabel("Total")
    
    plt.show()

# Column Bi_2 variable done,NUMERICAL - CATEGORICAL DONE 
#
#   END OF THE CATEGORICAL D
# 

# Column Bi_2 variable , NUMERICAL - NUMERICAL  
#
#   START OF THE NUMERICAL - NUMERICAL 
# 
""" 
popularity: The popularity of a track is a value between 0 and 100, with 100 being the most popular.
duration_ms: The track length in milliseconds
explicit: Whether or not the track has explicit lyrics (true = yes it does; false = no it does not OR unknown)
danceability: Describes how suitable a track is for dancing A value between 0.0 and 1.0.
energy: Energy is a measure from 0.0 to 1.0 and represents a perceptual measure of intensity and activity.
key: The key the track is in. Integers map to pitches using standard Pitch Class notation. E.g. 0 = C, 1 = C♯/D♭, 2 = D, and so on. If no key was detected, the value is -1
loudness: The overall loudness of a track in decibels (dB)
mode: Indicates the modality (major or minor) of a track. Major is represented by 1 and minor is 0
speechiness: Speechiness detects the presence of spoken words in a track. Value Between 0 and 1.
acousticness: A confidence measure from 0.0 to 1.0 of whether the track is acoustic. 1.0 represents high confidence the track is acoustic
instrumentalness: Predicts whether a track contains no vocals. The closer the instrumentalness value is to 1.0, the greater likelihood the track contains no vocal content
liveness: Detects the presence of an audience in the recording. A value between 0.0 and 1.0.
valence: A measure from 0.0 to 1.0 describing the musical positiveness conveyed by a track.
tempo: The overall estimated tempo of a track in beats per minute (BPM).
time_signature: A notational convention to specify how many beats are in each bar (or measure).
track_genre: The genre in which the track belongs"""
N_FOR_DENSE : int = 5000

NUMERIC_COLUMNS : List[int] = [
    DANCEABILITY, 
    LOUDNESS,
    SPEECHINESS,
    INSTRUMENTALNESS,
    ACOUSTICNESS,
    LIVENESS,
    VALENCE
    
]

SCALER_DANCEABILITY : MinMaxScaler = MinMaxScaler()
SCALER_LOUDNESS : MinMaxScaler = MinMaxScaler()
SCALER_SPEECHINESS : MinMaxScaler = MinMaxScaler()
SCALER_INSTRUMENTALNESS :MinMaxScaler = MinMaxScaler()
SCALER_ACOUSTICNESS : MinMaxScaler = MinMaxScaler()
SCALER_LIVENESS : MinMaxScaler = MinMaxScaler()
SCALER_VALENCE : MinMaxScaler = MinMaxScaler()

IS_SCALER_DANCEABILITY_DONE : bool = False
IS_SCALER_LOUDNESS_DONE : bool = False
IS_SCALER_SPEECHINESS_DONE : bool = False
IS_SCALER_INSTRUMENTALNESS_DONE : bool = False
IS_SCALER_ACOUSTICNESS_DONE : bool = False
IS_SCALER_LIVENESS_DONE : bool = False
IS_SCALER_VALENCE_DONE : bool = False

P_VALUE_THRESHOLD : float = 0.05
RANDOM_STATE : int = 0

BINS_1000 : int = 1000
BINS_100 : int = 100
def Bi_PrintScatterPlotDensity(
    col_another_x : str, 
    col_another_y : str,
    title_plot_1 : str = None,
):
    title_plot_1 = f"Density Plot for {col_another_x} and {col_another_y}"
    plt.figure(figsize=(12, 8), dpi=100)
    x : np.ndarray = df.loc[:N_FOR_DENSE, col_another_x].values
    y : np.ndarray = df.loc[:N_FOR_DENSE, col_another_y].values
    sns.scatterplot(x=x, y=y)
    sns.kdeplot(x=x, y=y, levels=6, alpha=0.6, fill=True, color="blue")
    plt.title(title_plot_1)
    plt.ylabel(col_another_y)
    plt.xlabel(col_another_x)
    plt.show()

def Bi_PrintHeatmap(dfCorr : pd.DataFrame, algorithm :str):
    plt.figure(figsize=(12,7))
    matrixTriu = np.triu(dfCorr)
    sns.heatmap(dfCorr, cmap="BuGn", annot=True, cbar=False,mask=matrixTriu)
    plt.title(f"Heatmap for Numeric variables ${algorithm}")
    plt.show()

def Bi_MannWhitneyTest(
    col_x : str ,
    col_y : str ,
    dfLocal : pd.DataFrame
    
):  
    statistic, p_value = stats.mannwhitneyu(dfLocal[col_x], dfLocal[col_y])
    sns.histplot(data=dfLocal, x=col_x, y=col_y, kde=True, bins=BINS_100)
    plt.title("Histogram Distribution bins %s plot for %s And %s " %(BINS_100, col_x, col_y))
    plt.show()
    print("Mann-Whitney Test for %s & %s" %(col_x, col_y))
    print("Statistics : %f " %(statistic))
    print("P-Value : %f" % (p_value))
    if(p_value <= P_VALUE_THRESHOLD):
        print("There is Statistically Significance difference between group %s and group %s" % (col_x, col_y))
    else :
        print("There is NO Statistically Significance difference between group %s and group %s" % (col_x, col_y))


def Bi_SimpleLinearRegression(
    x_var : str,
    y_var : str,
    dfLocal : pd.DataFrame
):
    Bi_MannWhitneyTest(x_var, y_var, dfLocal=dfLocal)
    print(f"Linear regression for feature {x_var} & {y_var}")
    x : np.ndarray = dfLocal[x_var].values.reshape(-1,1)
    y : np.ndarray = dfLocal[y_var].values
    linearRegression : LinearRegression = LinearRegression()
    x_preprocessed, y_preprocessed, x_val, y_val = Bi_PreprocessingInputData(x, y, x_var )
    linearRegression.fit(x_preprocessed, y_preprocessed)
    
    y_predicted : np.ndarray = linearRegression.predict(x_val)
    Bi_PlotPredictedAndReal(y_val, y_predicted, x_val, x_var, y_var)
    
    Bi_PrintLinRegPerformance(linearRegression, y_val , y_predicted)

    Bi_PrintResidualPlot(y_val, y_val - y_predicted, "Residual Plot of --%s-- as Target Variable (X : %s)" % (y_var, x_var), y_var)

def Bi_MaptoScaler(columnName : str , x_value : np.ndarray):
    x_copy : np.ndarray = x_value
    global DANCEABILITY, IS_SCALER_DANCEABILITY_DONE
    global LOUDNESS, IS_SCALER_LOUDNESS_DONE
    global SPEECHINESS, IS_SCALER_SPEECHINESS_DONE
    global LIVENESS, IS_SCALER_LIVENESS_DONE
    global VALENCE, IS_SCALER_VALENCE_DONE
    global ACOUSTICNESS, IS_SCALER_ACOUSTICNESS_DONE
    global INSTRUMENTALNESS, IS_SCALER_INSTRUMENTALNESS_DONE

    if(columnName == DANCEABILITY and not IS_SCALER_DANCEABILITY_DONE ):
        x_copy = SCALER_DANCEABILITY.fit_transform(x_value)
        IS_SCALER_DANCEABILITY_DONE = True
    elif(columnName == LOUDNESS and not IS_SCALER_LOUDNESS_DONE):
        x_copy = SCALER_LOUDNESS.fit_transform(x_value)
        IS_SCALER_LOUDNESS_DONE = True
    elif(columnName == SPEECHINESS and not IS_SCALER_SPEECHINESS_DONE):
        x_copy = SCALER_SPEECHINESS.fit_transform(x_value)
        IS_SCALER_SPEECHINESS_DONE = True
    elif(columnName == LIVENESS and not IS_SCALER_LIVENESS_DONE):
        x_copy = SCALER_LIVENESS.fit_transform(x_value)
        IS_SCALER_LIVENESS_DONE = True
    elif(columnName == VALENCE and not IS_SCALER_VALENCE_DONE):
        x_copy = SCALER_VALENCE.fit_transform(x_value)
        IS_SCALER_VALENCE_DONE = True
    elif(columnName == INSTRUMENTALNESS and not IS_SCALER_INSTRUMENTALNESS_DONE):
        x_copy = SCALER_INSTRUMENTALNESS.fit_transform(x_value)
        IS_SCALER_INSTRUMENTALNESS_DONE = True
    elif(columnName == ACOUSTICNESS and not IS_SCALER_ACOUSTICNESS_DONE):
        x_copy = SCALER_ACOUSTICNESS.fit_transform(x_value)
        IS_SCALER_ACOUSTICNESS_DONE = True
        
    ## ELIF FOR Transform Only
    elif(columnName == DANCEABILITY and IS_SCALER_DANCEABILITY_DONE):
        x_copy = SCALER_DANCEABILITY.transform(x_value)
    elif(columnName == LOUDNESS and IS_SCALER_LOUDNESS_DONE):
        x_copy = SCALER_LOUDNESS.transform(x_value)
    elif(columnName == SPEECHINESS and IS_SCALER_SPEECHINESS_DONE):
        x_copy = SCALER_SPEECHINESS.transform(x_value)
    elif(columnName == LIVENESS and IS_SCALER_LIVENESS_DONE):
        x_copy = SCALER_LIVENESS.transform(x_value)
    elif(columnName == VALENCE and IS_SCALER_VALENCE_DONE):
        x_copy = SCALER_VALENCE.transform(x_value)
    elif(columnName == INSTRUMENTALNESS and IS_SCALER_INSTRUMENTALNESS_DONE):
        x_copy = SCALER_INSTRUMENTALNESS.transform(x_value)
    elif(columnName == ACOUSTICNESS and IS_SCALER_ACOUSTICNESS_DONE):
        x_copy = SCALER_ACOUSTICNESS.transform(x_value)
    return x_copy
    
def Bi_PreprocessingInputData(x : np.ndarray, y : np.ndarray, column_x : str):
    x_scaled = Bi_MaptoScaler(column_x, x)
    x_train, x_val, y_train, y_val  = train_test_split(x_scaled, y , random_state= RANDOM_STATE, test_size=0.2)

    return x_train, y_train, x_val, y_val

def Bi_PlotPredictedAndReal(y_true : np.ndarray, y_pred : np.ndarray, x_val : np.ndarray,
                            x_col : str, y_col : str):
    plt.plot(x_val, y_pred, label="Predicted")   
    sns.histplot(x=x_val.squeeze(-1), y=y_true, bins=BINS_100, color="red", label="observerable")
    plt.title("Predicted variables ")
    plt.ylabel(y_col)
    plt.xlabel(x_col)
    plt.legend()
    plt.show()

def Bi_PrintLinRegPerformance(linearRegression : LinearRegression , y_val : np.ndarray, y_predicted : np.ndarray):
    print("Linear Regression performance\n")
    print(f"Coef : {linearRegression.coef_}, Intercept : {linearRegression.intercept_}")
    print(f"MSE : {mean_squared_error(y_val, y_predicted)}")
    print(f"RMSE : {np.sqrt(mean_squared_error(y_val, y_predicted))}")
    
def Bi_PrintResidualPlot(y_true : np.ndarray, residual : np.ndarray, title : str, y_col : str):
    zeroLine :np.ndarray =np.zeros_like(residual)
    
    plt.figure(figsize=(10,5))
    plt.scatter(y_true, residual,label="Residual")
    plt.plot(y_true, zeroLine, color = "red", linestyle="--")
    randomizeYTrue  : np.ndarray  = np.random.choice(y_true, size=N_FOR_DENSE)
    randomizeYResidual : np.ndarray = np.random.choice(residual, size=N_FOR_DENSE)
    sns.histplot(x=y_true, y=residual, kde=True, bins=BINS_1000)
    sns.kdeplot(x=randomizeYTrue,y=randomizeYResidual, fill=True, color="blue", alpha=0.6, levels=7) 
    plt.ylabel(y_col)
    plt.xlabel(y_col)
    plt.title(title)
    plt.show()

def QQPlot(values : np.ndarray):
    sm.qqplot(values, line="45")
    plt.show()


#Bi_SimpleLinearRegression(DANCEABILITY, LOUDNESS, dfLocal= df)
#Bi_MannWhitneyTest(DANCEABILITY, LOUDNESS, dfLocal=df)

#%% CLUSTERING FOR NUMERICAL VALUES ini nanti dikelarin
DEFAULT_K_CLUSTERING : int = 5
COLUMN_CLUSTERING_TEST  : List[str] = [
    DANCEABILITY,
    INSTRUMENTALNESS,
    LIVENESS
]

KMEANS_KWARGS : Dict[str, any] = {
    "init" : "random",
    "n_init" : 10,
    "max_iter" : 300,
    "random_state" : SEED
}
MAX_CENTROID : int = 10
def findBestK_ElbowMethod(
    columnList : List[str],
    dfLocal : pd.DataFrame
):
    x_normalized : pd.DataFrame = pd.DataFrame(columns=columnList)
    sse_k : List = []
    x : pd.DataFrame = dfLocal[columnList]
    for column in columnList:
        x_norm : np.ndarray = Bi_MaptoScaler(column, x[column].values.reshape(-1,1)).squeeze()
        x_normalized[column] = x_norm
    x_range = range(1, MAX_CENTROID + 1)
    for k in x_range:
        kmeans : KMeans = KMeans(n_clusters= k,**KMEANS_KWARGS)
        kmeans.fit(x_normalized)
        sse_k.append(kmeans.inertia_)
    title = "Elbow method for %s : " % (" ".join([col for col in columnList]))
    plt.plot(x_range, sse_k)
    plt.xticks(x_range)
    plt.title(title)
    plt.show()

def findBestK_SilhouetteMethod(
    columnList : List[str],
    dfLocal : pd.DataFrame
):
    x_normalized : pd.DataFrame = pd.DataFrame(columns=columnList)
    sse_k : List = []
    x : pd.DataFrame = dfLocal[columnList]
    for column in columnList:
        x_norm : np.ndarray = Bi_MaptoScaler(column, x[column].values.reshape(-1,1)).squeeze()
        x_normalized[column] = x_norm
    x_range = range(2, 4)
    for k in x_range:
        print("Check for NCluster : %s " %(k))
        kmeans : KMeans = KMeans(n_clusters= k,**KMEANS_KWARGS)
        kmeans.fit(x_normalized)
        #sse_k.append(silhouette_score(x_normalized, kmeans.labels_))
    title = "Silhouette method for %s : " % (" ".join([col for col in columnList]))
    plt.plot(x_range, sse_k)
    plt.xticks(x_range)
    plt.title(title)
    plt.show()

#findBestK_ElbowMethod(COLUMN_CLUSTERING_TEST, df)
#findBestK_SilhouetteMethod(COLUMN_CLUSTERING_TEST, df) Jangan dipake ngeleg, tasknya banyak
SCATTER_3D_MODES_PLOTLY : str = "markers"
SCATTER_3D_DEFAULT_OPACITY : float = 0.05

DBSCAN_epsilon : float = 0.05
DBSCAN_min_samples : int = 50
def Plot_3D_Plotly(
    x : np.ndarray,
    y : np.ndarray,
    z : np.ndarray,
    columnList : List[str],
    labels : np.ndarray = None,
    isCentroidExist: bool = False,
    centroids : np.ndarray = None
    
):
    fig : go.Figure = go.Figure(
        data = go.Scatter3d(
        x=x,
        y=y,
        z=z,
        mode=SCATTER_3D_MODES_PLOTLY,
        marker = {
            "size" : 3,
            "color" : labels,
            "colorscale" : "Viridis",
            "opacity" : SCATTER_3D_DEFAULT_OPACITY,
        },
        text=[f"Cluster {label} " for label in labels],
        showlegend=True,
        )
    )
    
    if(isCentroidExist):
        fig.add_trace(go.Scatter3d(
            x=centroids[:, 0],
            y=centroids[:, 1],
            z=centroids[:, 2],
            mode=SCATTER_3D_MODES_PLOTLY,
            marker= {
                "size" : 5,
                "color" : "red",
                "symbol" : "x"
            },
        ))

    fig.update_layout(
        title="3D Clustering Kmeans with %s \n" % ("<br>".join([col for col in columnList])),
        scene={ "xaxis_title" : columnList[0],
            "yaxis_title" : columnList[1],
            "zaxis_title" : columnList[2]
        }
    )
    fig.show()
    
def Bi_3_All_ClusteringKmeans(
    columnList : List[str],
    dfLocal : pd.DataFrame
):
    x_normalized : pd.DataFrame = pd.DataFrame(columns=columnList)
    x : pd.DataFrame = dfLocal[columnList]
    for column in columnList:
        x_norm : np.ndarray = Bi_MaptoScaler(column, x[column].values.reshape(-1,1)).squeeze()
        x_normalized[column] = x_norm
    
    kmeans : KMeans = KMeans(n_clusters=DEFAULT_K_CLUSTERING)
    kmeans.fit(x_normalized)
    x_norm_values : np.ndarray = x_normalized.iloc[:,0].values
    y_norm_values : np.ndarray = x_normalized.iloc[:, 1].values
    z_norm_values : np.ndarray = x_normalized.iloc[:, 2].values
    centroids : np.ndarray = kmeans.cluster_centers_
    Plot_3D_Plotly(x_norm_values,
                   y_norm_values,
                   z_norm_values,
                   columnList,
                   kmeans.labels_,
                   True,
                   centroids)

def Bi_3_All_ClusteringDBSCAN(
    columnList : List[str],
    dfLocal : pd.DataFrame
):
    x_normalized : pd.DataFrame = pd.DataFrame(columns=columnList)
    x : pd.DataFrame = dfLocal[columnList]
    for column in columnList:
        x_norm : np.ndarray = Bi_MaptoScaler(column, x[column].values.reshape(-1,1)).squeeze()
        x_normalized[column] = x_norm
    
    x_norm_values : np.ndarray = x_normalized.iloc[:, 0].values
    y_norm_values : np.ndarray = x_normalized.iloc[:, 1].values
    z_norm_values : np.ndarray = x_normalized.iloc[:, 2].values 

    dbScan : DBSCAN = DBSCAN(
        eps=DBSCAN_epsilon,
        min_samples=DBSCAN_min_samples
        ).fit(X=x_normalized)
    Plot_3D_Plotly(
        x_norm_values,
        y_norm_values,
        z_norm_values,
        columnList,
        dbScan.labels_,
        False,
        None,
    )


def Bi_All_ClusteringKMeans(
    columnList : List[str],
    dfLocal : pd.DataFrame, 
):
    
    x_normalized : pd.DataFrame = pd.DataFrame(columns=columnList)
    
    x : pd.DataFrame = dfLocal[columnList]
    for column in columnList:
        x_norm : np.ndarray = Bi_MaptoScaler(column, x[column].values.reshape(-1, 1)).squeeze()
        x_normalized[column] = x_norm

    pca_obj : PCA = PCA(n_components=2 , random_state= SEED)
    
    normalized_pca_dataset : np.ndarray = pca_obj.fit_transform(x_normalized)
    kmeans : KMeans = KMeans(n_clusters=DEFAULT_K_CLUSTERING, 
                             **KMEANS_KWARGS)
    kmeans.fit(normalized_pca_dataset)
    
    step : float = 0.02
    ## + 1 & -1 just for making sure all data points included within the grid
    x_min, x_max = normalized_pca_dataset[:, 0].min() - 1, normalized_pca_dataset[:, 0].max() + 1
    y_min, y_max = normalized_pca_dataset[:, 1].min() - 1, normalized_pca_dataset[: , 1].max() + 1

    # THis just trying to make new dataset
    x_arange : np.ndarray = np.arange(x_min, x_max, step)
    y_arange : np.ndarray = np.arange(y_min, y_max, step) 
    xx, yy = np.meshgrid(x_arange, y_arange)
    
    # Predict the cluster for each pooint
    x_flattened : np.ndarray = xx.ravel()
    y_flattened : np.ndarray = yy.ravel()
    
    # This just like adding columns ( Making n2DArray )
    z : np.ndarray = kmeans.predict(np.c_[x_flattened, y_flattened])
    z : np.ndarray = z.reshape(xx.shape) # Because we got the flattened, we need to 

    fig : go.Figure = go.Figure()
    
    fig.add_traces(
        go.Heatmap(
            x=np.linspace(x_min, x_max, xx.shape[0]), # Create dummy x with total xx.shape[0] 
            y=np.linspace(y_min, y_max, xx.shape[1]),
            z=z,
            showscale=False,
            colorscale="Viridis"
        )
    )
    
    fig.add_traces(
        go.Scatter(
            x = kmeans.cluster_centers_[:, 0],
            y = kmeans.cluster_centers_[:, 1],
            mode="markers",
            marker= {
                "color": "black",
                "size" : 15,
                "symbol" : "x",
                "line" : {
                    "color" : "black",
                    "width" : 3
                }
            }
        )
    )
    
    fig.update_layout(
    title="K-means clustering on All Numerical data",
    xaxis=dict(title="First Principal Component"),
    yaxis=dict(title="Second Principal Component"),
)

    fig.show()

def Bi_All_DBScan(
    columnList : List[str],
    dfLocal : pd.DataFrame,
    
):
    
    
    x_normalized : pd.DataFrame = pd.DataFrame(columns = columnList)
    x : np.ndarray = df[columnList]
    for column in columnList:
        x_norm : np.ndarray = Bi_MaptoScaler(column, x[column].values.reshape(-1,1))
        x_normalized[column] = x_norm.squeeze()
    
    pca_obj : PCA = PCA(n_components=2, random_state=SEED)
    pca_new_dataset : np.ndarray = pca_obj.fit_transform(x_normalized)[:N_FOR_DENSE]
    
    dbScan : DBSCAN = DBSCAN(eps= DBSCAN_epsilon,
                             min_samples=DBSCAN_min_samples,
                             n_jobs=-1)
    dbScan.fit(pca_new_dataset)
    
    knn : KNeighborsClassifier = KNeighborsClassifier(n_neighbors=len(np.unique(dbScan.labels_)))    
    knn.fit(pca_new_dataset, dbScan.labels_)
    step : int =  0.02

    x_min, x_max = pca_new_dataset[:, 0].min() - 1 , pca_new_dataset[:, 0].max() + 1
    y_min, y_max = pca_new_dataset[:, 1].min() - 1 , pca_new_dataset[:, 1].max() + 1
    
    x_arange : np.ndarray = np.arange(x_min, x_max, step)
    y_arange : np.ndarray = np.arange(y_min, y_max, step)
    
    xx, yy = np.meshgrid( x_arange, y_arange)
    
    predicted : np.ndarray = knn.predict(np.c_[xx.ravel(), yy.ravel()])
    predicted = predicted.reshape(xx.shape)
    
    fig : go.Figure = go.Figure()
    
    fig.add_trace(
        go.Heatmap(
            x=np.arange(x_min, x_max, predicted.shape[0]),
            y=np.arange(y_min, y_max, predicted.shape[1]),
            z=predicted,
            showscale=False,
            colorscale="viridis"
        )
    )
    
    fig.update_layout(
        title="DBScan clustering on All Numerical data",
        xaxis=dict(title="First Principal Component"),
        yaxis=dict(title="Second Principal Component"),
    )
    
    
    fig.show()
    
Bi_All_DBScan(
    COLUMN_CLUSTERING_TEST,
    df
)  
    
Bi_All_ClusteringKMeans(COLUMN_CLUSTERING_TEST,df)     
    
    
#%%

#Bi_3_All_ClusteringDBSCAN(COLUMN_CLUSTERING_TEST, df)


#%%
Bi_PrintHeatmap(df.corr("kendall"), "Kendall")
Bi_PrintHeatmap(df.corr("pearson"), "Pearson")

#%%
chosen : str = DANCEABILITY
for column in NUMERIC_COLUMNS:
    if(column != chosen):
        Bi_PrintScatterPlotDensity(chosen, column,"YES")
        break

#%%%%%%%%%

Bi_PrintInfoInstrumentalness(
    ACOUSTICNESS, ARTISTS,
    f"Top {TOP_POPULARITY} {ARTISTS} by popularity ",
    "","Total",TOP_POPULARITY
)
#%%
Bi_PrintInfoInstrumentalness(
    INSTRUMENTALNESS, ARTISTS,
    f"Top {TOP_POPULARITY} {ARTISTS} by popularity ",
    "","Total",TOP_POPULARITY
)
#%%
Bi_PrintInfoSpeechiness(
    SPEECHINESS, ARTISTS,
    f"Top {TOP_POPULARITY} {ARTISTS} by popularity ",
    "","Total",TOP_POPULARITY
)
#%% 
Bi_PrintInfoLiveness(
    LIVENESS, ARTISTS,
    f"Top {TOP_POPULARITY} {ARTISTS} by popularity ",
    "","Total",TOP_POPULARITY
)

#%%   
Bi_PrintInfoDanceability(
    DANCEABILITY, ARTISTS,
    f"Top {TOP_POPULARITY} {ARTISTS} by popularity ",
    "","Total",TOP_POPULARITY
)

    
    
#%%
#%%
  
Bi_PrintInfoLoudness(
    LOUDNESS,ARTISTS,
    f"Top {TOP_POPULARITY} {ARTISTS} by popularity ",
    "","Total",TOP_POPULARITY
) 
    
    


#%%    
for column in POPULARITY_COMPARABLE_VALUES:
    Bi_PrintInfoPopularity(
        POPULARITY,column,
        f"Top {TOP_POPULARITY} {column} by popularity ",
        "","Total",TOP_POPULARITY
    )



#%%    
    
#%% 
PrintMoreInfoAboutAlbum()
#%%    
PrintMoreInfoAboutArtist()

#%%
#%% Code Execution
PrintColumnsDescription()
#%%
#%%
PrintColumnTypes()
#%%
PrintColumnsNull()

#%%

PrintDuplicatedInfo()