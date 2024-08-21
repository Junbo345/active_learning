import logging
from warnings import simplefilter
from dataclasses import dataclass, field
from omegaconf import OmegaConf
import numpy as np
import pandas as pd
import active_learnig_badge as alb


def get_data(iterations, initial, batch, method, data, cfg):
    """
    Generate data for plotting the performance of different sampling methods over iterations.

    Parameters:
    iterations (list[int]): Number of iterations for each method
    initial (list[int]): Initial pool of data for each method
    batch (list[int]): Number of data points in each batch of iteration
    method (list[str]): Regression method for each iteration
    """
    col = max(iterations)

    cfg = cfg
    test = data.sample(n=1500, random_state=1)
    train = data.drop(test.index).reset_index(drop=True)
    test.reset_index(drop=True)
    output = pd.DataFrame()

    for ind in range(len(batch)):
        cfg.loop.times = iterations[ind]
        cfg.loop.init = initial[ind]
        cfg.loop.batch_size = batch[ind]
        cfg.learner.regression_method = method[ind]

        # for a quick baseline check
        # print(get_y_baseline(train, test, cfg))
        # continue

        # Initialize the X-axis values for the plot
        # this is now number of labelled points (galaxies)
        x = np.arange(cfg.loop.times + 1) * cfg.loop.batch_size + cfg.loop.init

        # Calculate the Y-axis values for different sampling methods
        y1 = abs(np.log(1 - np.array(alb.badge(train, test, cfg))))
        y2 = abs(np.log(1 - np.array(alb.diverse_tree(train, test, cfg))))
        y3 = abs(np.log(1 - np.array(alb.randomapp(train, test, cfg))))
        output[f'x{ind + 1} {method[ind]}'] = np.concatenate([x, np.full(col - len(x) + 1, np.nan)])
        output[f'badge_{ind + 1}'] = np.concatenate([y1, np.full(col - len(x) + 1, np.nan)])
        output[f'uncertainty_{ind + 1}'] = np.concatenate([y2, np.full(col - len(x) + 1, np.nan)])
        output[f'random_{ind + 1}'] = np.concatenate([y3, np.full(col - len(x) + 1, np.nan)])

    # Save the output to a CSV file
    output.to_csv('model.csv', index=False)


if __name__ == '__main__':
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

    # loaddata()
    get_data(iterations=[6, 10], initial=[50, 70], batch=[500, 300], method=["pytorch_N", "pytorch_N"], data=data,
             cfg=cfg)
    # get_data(iterations=[75], initial=[50], batch=[30], method=["pytorch_N"])
