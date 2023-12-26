import DataPreps
import mlflow
from hpsklearn import HyperoptEstimator
from omegaconf import OmegaConf
from typing import Union

from Main_Enum import ModelName

def run_hyper_params(config,
                     modelName: ModelName,
                     target: str
                     ):
    x, y = DataPreps.run(config, target)

if __name__ == "__main__":
    ...
