from dataclasses import dataclass, field
from omegaconf import OmegaConf
import logging
from warnings import simplefilter
import pandas as pd
import data_calculator


@dataclass
class LoopConfig:
    times: int = 50  # Number of iterations in the active learning loop
    init: int = 50  # Initial number of samples to start the active learning process
    batch_size: int = 15  # Number of samples to add in each iteration


@dataclass
class Learner:
    regression_method: str = 'M'  # Default regression method (L: Logistic Regression, M: MLPClassifier)


@dataclass
class ActiveLearningConfig:
    loop: LoopConfig = field(default_factory=LoopConfig)
    learner: Learner = field(default_factory=Learner)
    feature_cols: list = field(
        default_factory=lambda: ['feat_pca_{}'.format(i) for i in range(40)])  # Feature column names
    label_cols: list = field(
        default_factory=lambda: ['label', 'label2'])  # Label column names


cfg = OmegaConf.structured(ActiveLearningConfig)

# https://lightning.ai/docs/pytorch/stable/extensions/logging.html#configure-console-logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

# ignore all warnings
simplefilter(action='ignore')

data = pd.read_csv("strong_lense_train.csv")

data_calculator.get_data(iterations=[7], initial=[50], batch=[200], method=["pytorch_N"],
                         data=data, cfg=cfg)
