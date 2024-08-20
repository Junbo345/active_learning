# **Active Learning Strategies for Euclid Space Telescope**


This code contains different active learning strategies to improve the classification rate for galaxy morphology.

# Installation

To Download the code using git: ```git@github.com:Junbo345/active_learning.git``` <br/>
<br/>
Then to install pytorch necessary packages into your 

To Download the code using GithubCLI to bring GitHub to your terminal: ```gh repo clone Junbo345/active_learning```

# Quickstart

Suppose you wanted to determine how well the uncertainty method can accurately classify lrg galaxies in some small dataset that contains a number of lrg and non-lrg galaxies. We can calculate the proposed model scores by: <br/>

```
@dataclass
class LoopConfig:
    times: int=100
    init: int=50
    batch_size: int=15

@dataclass
class Learner:
    regression_method: str='L'

@dataclass
class ActiveLearningConfig:
    loop: LoopConfig = field(default_factory=LoopConfig)
    learner: Learner = field(default_factory=Learner)
    feature_cols: list = field(default_factory=lambda: ['feat_pca_{}'.format(i) for i in range(20)])
    label_cols: list = field(default_factory=lambda: ['s1_lrg_fraction'])```
