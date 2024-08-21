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
    regression_method: str = 'L'  # Default regression method (L: Logistic Regression, M: MLPClassifier)


@dataclass
class ActiveLearningConfig:
    loop: LoopConfig = field(default_factory=LoopConfig)
    learner: Learner = field(default_factory=Learner)
    feature_cols: list = field(
        default_factory=lambda: ['feat_pca_{}'.format(i) for i in range(20)])  # Feature column names
    label_cols: list = field(
        default_factory=lambda: ['s1_lrg_fraction', 's1_spiral_fraction', 'other'])  # Label column names


cfg = OmegaConf.structured(ActiveLearningConfig)

# https://lightning.ai/docs/pytorch/stable/extensions/logging.html#configure-console-logging
logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

# ignore all warnings
simplefilter(action='ignore')

data = pd.read_csv('model_cleaned.csv')


data_calculator.get_data(iterations=[7, 12], initial=[50, 70], batch=[500, 300], method=["pytorch_N", "pytorch_N"], data=data, cfg=cfg)