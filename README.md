# **Active Learning Strategies for Euclid Space Telescope**


This code contains different active learning strategies to improve the classification rate for galaxy morphology.

# Installation

To Download the code using git: ```git@github.com:Junbo345/active_learning.git``` <br/>
<br/>
Then to install pytorch necessary packages into your 

To Download the code using GithubCLI to bring GitHub to your terminal: ```gh repo clone Junbo345/active_learning```

# Quickstart

Suppose you wanted to determine how well some query method can accurately classify between lrg galaxies, ring galaxies, and other galaxies in some small dataset. We can calculate the proposed model scores by first configurating the number of iterations of batch size of the query function and determining the regression method: <br/>

```ruby
import numpy as np
import active_learnig_badge as al
import estimators_badge.py

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

# load the dataset using provided csv file
data = al.loaddata(cfg)

# generate a csv file containing the model scores of the different query methods
al.get_data(iterations = [50, 60], initial = [30, 40], batch = [50, 60], method = ["pytorch_N","pytorch_N"])
```
# Dataset

The dataset that was used for this project contained ~5200 galaxies. It can be viewed on excel using [This Link](https://artatuoft.slack.com/files/U05QNJ61FA7/F07CV51CFFG/karina_representations_for_junbo_khalid.csv).







