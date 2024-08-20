# **Active Learning Strategies for Euclid Space Telescope**


This code contains different active learning strategies to improve the classification rate for galaxy morphology.

# Installation

To Download the code using git: ```git@github.com:Junbo345/active_learning.git``` <br/>
<br/>
Then to install pytorch necessary packages into your 

To Download the code using GithubCLI to bring GitHub to your terminal: ```gh repo clone Junbo345/active_learning```

# Quickstart

Suppose you wanted to determine how well some query method can accurately classify lrg galaxies in some small dataset that contains a number of lrg and non-lrg galaxies. We can calculate the proposed model scores by first configurating the number of iterations of batch size of the query function and determining the regression method: <br/>

```ruby
import numpy as np
import pandas as pd
import torch
from dataclasses import dataclass, field
from omegaconf import OmegaConf
from sklearn.neural_network import MLPClassifier
from sklearn.linear_model import LogisticRegression

@dataclass
class LoopConfig:
    times: int=100
    init: int=50
    batch_size: int=15

@dataclass
class Learner:
    regression_method: str='L' # Default regression method (L: Logistic Regression, M: MLPClassifier)

@dataclass
class ActiveLearningConfig:
    loop: LoopConfig = field(default_factory=LoopConfig)
    learner: Learner = field(default_factory=Learner)
    feature_cols: list = field(default_factory=lambda: ['feat_pca_{}'.format(i) for i in range(20)])
    label_cols: list = field(default_factory=lambda: ['s1_lrg'])
```
<br/>
Then loading the dataset: <br/>

```ruby
    expected_csv_loc = '/your/path/some_labelled_galaxies.csv'  
    df = pd.read_csv(expected_csv_loc)

    columns_to_select = cfg.feature_cols + cfg.label_cols
    df_s = df[columns_to_select]

    label_columns_expanded = df_s[cfg.label_cols].apply(label_highest_prob, axis=1, result_type='expand')

    label_columns_expanded.columns = cfg.label_cols

    df_final = pd.concat([df_s[cfg.feature_cols], label_columns_expanded], axis=1)
```
<br/>
Finally, compute a list of the model scores to identify how well your model classifies the data:
<br/>

```ruby
    test = loaddata(cfg).sample(n=1500, random_state=1)
    train = loaddata(cfg).drop(test.index).reset_index(drop=True)
    test.reset_index(drop=True)

    clf = get_active_learner(cfg)
    labelled = train.sample(n=cfg.loop.init, random_state=1)
    pool = train.drop(labelled.index)
    test_x = test[cfg.feature_cols].values
    test_y = test[cfg.label_cols].values
    trace = []
    X = labelled[cfg.feature_cols].values
    y = labelled[cfg.label_cols].values
    clf.fit(X, y)
    trace.append(clf.score(test_x, test_y))

    for iteration_n in range(cfg.loop.times):
        new = pool.sample(n=cfg.loop.batch_size)
        labelled = pd.concat([labelled, new])
        pool = pool.drop(new.index)
        X = labelled[cfg.feature_cols].values
        y = labelled[cfg.label_cols].values
        clf.fit(X, y)
        trace.append(clf.score(test_x, test_y))
    return trace
```




