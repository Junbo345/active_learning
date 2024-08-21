# **Active Learning Strategies for Euclid Space Telescope**

This code contains different active learning strategies to improve the classification rate for galaxy morphology. It contains 3 different acquisition functions (ambiguous, diverse, BADGE) that aim to maximize the model score using different regression methods such as an MLP or Multinomial Logistic Regression. After conducting multiple different experiments, it was found the BADGE method provided the highest model scores and was not affected by batch size changes meaning that it would provide the best classification for the Euclid data. 

# Installation

To Download the code using git: ```git@github.com:Junbo345/active_learning.git``` <br/>
<br/>
To Download the code using GithubCLI to bring GitHub to your terminal: ```gh repo clone Junbo345/active_learning```
<br/>

Otherwise, the code can be copied and pasted into a Python environment and run from there. <br/>

Other Python packages need to be installed: 
<br/>
Numpy <br/>
Pandas <br/>
Scikit-Learn <br/>
Torch <br/>
Torch-lightning <br/>
Omegaconf <br/>
Dataclasses <br/>
Warnings <br/>
Logging <br/>
Scipy <br/>

# Quickstart

Suppose you wanted to determine how well some query method can accurately classify between LRG galaxies, ring galaxies, and other galaxies in some small dataset. We can calculate the proposed model scores by first configuring the number of iterations, the initial batch, and the batch size of the query functions and determining the regression method. Then putting the calculated scores into a csv file: <br/>

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

# load the dataset using provided CSV file
data = al.loaddata(cfg)

# generate a CSV file containing the model scores of the different query methods
al.get_data(iterations = [50, 60], initial = [30, 40], batch = [50, 60], method = ["pytorch_N","pytorch_N"])
```
# Dataset

The dataset that was used for this project contained ~5200 galaxies. It can be viewed using [This Link](https://docs.google.com/spreadsheets/d/1wNmAqCF6vYWlkeholPEZQDJ1QFmoZ13O5fW1kR5rBoo/edit?gid=1126909556#gid=1126909556). 
<br/> It contains 20 feature columns and multiple target columns, each representing a possible galaxy shape. The s1_ and s2_ prefixes mean stage 1 and stage 2. All galaxies were labelled in stage 1. Only galaxies with at least one stage 1 vote for lens were labelled in stage 2. Hence, only a small portion of galaxies have stage 2 labels.







