from argparse import ArgumentParser
import ModelBuilder
import HyperparamsFinder
from Main_Enum import ModelName
from omegaconf import OmegaConf, DictConfig, ListConfig
from typing import List, Tuple

parser: ArgumentParser = ArgumentParser(
    prog="Spotify_analysis",
    description="train the model and show the feature importance ( support 1 model per run)"
)

config: Tuple[DictConfig, ListConfig] = OmegaConf.load('params.yaml')

supported_models: List[str] = [config.cmd.dt, config.cmd.rf,
                               config.cmd.knn, config.cmd.ada_bst,
                               config.cmd.lin_reg]

supported_targets: List[str] = [config.cmd.dance, config.cmd.energy,
                                config.cmd.loud, config.cmd.speech,
                                config.cmd.instrumental, config.cmd.live,
                                config.cmd.popularity, config.cmd.valence,
                                config.cmd.tempo, config.cmd.duration
                                ]
# Add a new argument for the model type
parser.add_argument('-tr', '--train', help="Train a model",
                    choices=supported_models, default=None)
parser.add_argument('-hyp', '--hyper',
                    help="Find optimal Hyperparams (Cannot be used along with --train)",
                    choices=supported_models, default=None)
parser.add_argument('-spt', '--split', help="number of split",
                    type=int, default=5)

parser.add_argument('-target', '--target',
                    help="Y value you would like to observe",
                    choices= supported_targets,
                    default=config.constant.popularity_str)

args = parser.parse_args()


def parseToModelName(modelName: str) -> ModelName:
    if (modelName == config.cmd.dt):
        return ModelName.DT
    elif (modelName == config.cmd.rf):
        return ModelName.RF
    elif (modelName == config.cmd.knn):
        return ModelName.KNN
    elif (modelName == config.cmd.ada_bst):
        return ModelName.ADA_BST
    elif (modelName == config.cmd.lin_reg):
        return ModelName.LIN_REG


def modelToRun(args):

    modelName: ModelName
    if (args.train is not None):
        modelName = parseToModelName(args.train)
    else:
        modelName = ModelName.LIN_REG
    ModelBuilder.run_models(config, modelName,
                            args.split, args.target)


if (args.train is not None):
    modelToRun(args)
elif (args.hyper is not None):
    ...
else:
    print("Model not supported yet")
    # Not implemented yet
