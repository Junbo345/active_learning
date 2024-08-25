# **Active Learning Strategies for Euclid Space Telescope**

This repository contains code for implementing various active learning strategies to enhance the classification accuracy of galaxy morphology using data from the Euclid Space Telescope. Three different acquisition functions are provided—Ambiguous, Diverse, and BADGE—that aim to maximize model performance through regression methods such as MLP (Multilayer Perceptron) and Multinomial Logistic Regression.

Through extensive experimentation, the BADGE method was found to consistently yield the highest model scores. Notably, BADGE is robust against changes in batch size, making it the most reliable strategy for classifying Euclid data. 

## Installation

To Download the code using git: ```git@github.com:Junbo345/active_learning.git``` <br/>
<br/>
To Download the code using GithubCLI to bring GitHub to your terminal: ```gh repo clone Junbo345/active_learning```
<br/>

Otherwise, the code can be copied and pasted into a Python environment and run from there. <br/>

### Required Python Packages

Ensure the following Python packages are installed:

- **Numpy**
- **Pandas**
- **Scikit-Learn**
- **Torch**
- **Torch-lightning**
- **Omegaconf**
- **Dataclasses**
- **Warnings**
- **Logging**
- **Scipy**

## File Overview

The key files in this repository are:

- **`user.py`**: Example script to run the project.
- **`active_learning_badge.py`**: Contains the active learning algorithms, including BADGE, diversity, uncertainty, and random (baseline).
- **`estimators_badge.py`**: Custom MLP models tailored for the BADGE method.
- **`data_cleaner.py`**: Script for cleaning the original galaxy data for use in BADGE training.
- **`data_calculator.py`**: Script to store the performance scores of each iteration.

Other files in the repository are not required for this project and can be ignored.

## Quickstart

Running the code will generate a file containing the performance scores for each iteration. The scores are calculated using the formula **|log(1-percent correct)|**. Below is a sample output:

![Sample Output](https://github.com/user-attachments/assets/29af6138-e814-4cbd-a672-e21a05b2d7b1)

### Data Preparation

To generate the data, you will need an initial dataset formatted as a DataFrame. The DataFrame should include:

- **Feature Columns**: Data for each feature.
- **Label Columns**: One-hot encoded labels for each class.

Sample data:

![Sample Data](https://github.com/user-attachments/assets/93abb92c-6e2a-4e16-b279-d27c4d7cead1)

The feature columns (e.g., `feat_pca_0` to `feat_pca_19`) represent the features of each sample, while the last three columns are the labels for each class.

Place this file in the same directory where the code is located. Then, navigate to the folder and modify the following lines in the `user.py` script: <br/>

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
<br/> It contains 20 feature columns and multiple target columns, each representing a possible galaxy shape. The s1_ and s2_ prefixes mean stage 1 and stage 2. All galaxies were labeled in stage 1. Only galaxies with at least one stage 1 vote for lens were labeled in stage 2. Hence, only a small portion of galaxies have stage 2 labels.







