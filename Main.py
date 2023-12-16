from argparse import ArgumentParser
import ModelBuilder
import HyperparamsFinder
from Main_Enum import ModelName
from typing import List

parser: ArgumentParser = ArgumentParser(
    prog="Spotify_analysis",
    description="train the model and show the feature importance ( support 1 model per run)"
)

supported_models: List[str] = ['decision_tree', 'random_forest',
                               'k_nearest_neighbors', 'ada_boost',
                               'linear_regression']

# Add a new argument for the model type
parser.add_argument('-tr', '--train', help="Train a model",
                    choices=supported_models, default=None)
parser.add_argument('-hyp', '--hyper',
                    help="Find optimal Hyperparams (Cannot be used with Train)",
                    choices=supported_models, default=None)
parser.add_argument('-spt', '--split', help="number of split",
                    type=int, default=5)

args = parser.parse_args()


def parseToModelName(modelName: str) -> ModelName:
    if (modelName == 'decision_tree'):
        return ModelName.DT
    elif (modelName == "random_forest"):
        return ModelName.RF
    elif (modelName == "k_nearest_neighbors"):
        return ModelName.KNN
    elif (modelName == "ada_boost"):
        return ModelName.ADA_BST
    elif (modelName == "linear_regression"):
        return ModelName.LIN_REG


def modelToRun(args):

    modelName: ModelName
    if (args.train is not None):
        modelName = parseToModelName(args.train)
    else:
        modelName = ModelName.LIN_REG
    ModelBuilder.run_models(modelName, args.split)


if (args.train is not None):
    modelToRun(args)
elif (args.hyper is not None):
    ...
else:
    print("Model not supported yet")
    # Not implemented yet
