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
**Numpy** <br/>
**Pandas** <br/>
**Scikit-Learn** <br/>
**Torch** <br/>
**Torch-lightning** <br/>
**Omegaconf** <br/>
**Dataclasses** <br/>
**Warnings** <br/>
**Logging** <br/>
**Scipy** <br/>

# Quickstart
The code will output a file that contains the performance scores of each iteration. The scores are calculated by **|log(1-percent correct)|**. Below is the sample output: <br/>

![image](https://github.com/user-attachments/assets/29af6138-e814-4cbd-a672-e21a05b2d7b1) <br/>

To generate the data, you need the initial data, which would be a data frame with feature columns containing data for each feature and label columns, which is the one-hot encoding of each class you want to categorize. Below is the sample data: <br/>

![image](https://github.com/user-attachments/assets/93abb92c-6e2a-4e16-b279-d27c4d7cead1)

Feat_pca 0-19 are features of each sample data, and the last three columns are labels for each class. <br/>

The file should be put in the same folder as the code is downloaded. Nevagate to the folder, Open users, replace the file name in the following code: <br/>

```ruby
data = pd.read_csv("model_cleaned.csv")
```
<br/>

Enter feature & label columns in the following code: <br/>

```ruby
    feature_cols: list = field(
        default_factory=lambda: ['feat_pca_{}'.format(i) for i in range(20)])  # Feature column names
    label_cols: list = field(
        default_factory=lambda: ['s1_lrg_fraction', 's1_spiral_fraction', 'other'])  # Label column names
```
<br/>

Enter Initial sample size, number of iterations, batch size, and model type you want to test as a list: <br/>
```ruby
data_calculator.get_data(iterations=[7, 12], initial=[50, 70], batch=[500, 300], method=["pytorch_N", "pytorch_N"],
                         data=data, cfg=cfg)
```
<br/>
Run the script, you will then see the file in the folder after it finishes

# Dataset

The dataset that was used for this project contained ~5200 galaxies. It can be viewed using [This Link](https://docs.google.com/spreadsheets/d/1wNmAqCF6vYWlkeholPEZQDJ1QFmoZ13O5fW1kR5rBoo/edit?gid=1126909556#gid=1126909556). 
<br/> It contains 20 feature columns and multiple target columns, each representing a possible galaxy shape. The s1_ and s2_ prefixes mean stage 1 and stage 2. All galaxies were labelled in stage 1. Only galaxies with at least one stage 1 vote for lens were labelled in stage 2. Hence, only a small portion of galaxies have stage 2 labels.







