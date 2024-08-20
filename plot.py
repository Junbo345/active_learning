import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Read the CSV file into a DataFrame
df = pd.read_csv('model_200.csv')
sns.set_context('talk')
# Calculate mean values
df['mean_badge'] = df[[f'badge_{i}' for i in range(1, 11)]].median(axis=1)
df['mean_unc'] = df[[f'uncertainty_{i}' for i in range(1, 11)]].median(axis=1)
df['mean_ran'] = df[[f'random_{i}' for i in range(1, 11)]].median(axis=1)

# Calculate 75th and 25th percentile values
df['percentile75_badge'] = df[[f'badge_{i}' for i in range(1, 11)]].apply(lambda x: np.percentile(x, 75), axis=1)
df['percentile25_badge'] = df[[f'badge_{i}' for i in range(1, 11)]].apply(lambda x: np.percentile(x, 25), axis=1)

df['percentile75_unc'] = df[[f'uncertainty_{i}' for i in range(1, 11)]].apply(lambda x: np.percentile(x, 75), axis=1)
df['percentile25_unc'] = df[[f'uncertainty_{i}' for i in range(1, 11)]].apply(lambda x: np.percentile(x, 25), axis=1)

df['percentile75_ran'] = df[[f'random_{i}' for i in range(1, 11)]].apply(lambda x: np.percentile(x, 75), axis=1)
df['percentile25_ran'] = df[[f'random_{i}' for i in range(1, 11)]].apply(lambda x: np.percentile(x, 25), axis=1)

# Calculate error bars: the difference between the 75th/25th percentiles and the mean
df['upper_error_badge'] = df['percentile75_badge'] - df['mean_badge']
df['lower_error_badge'] = df['mean_badge'] - df['percentile25_badge']

df['upper_error_unc'] = df['percentile75_unc'] - df['mean_unc']
df['lower_error_unc'] = df['mean_unc'] - df['percentile25_unc']

df['upper_error_ran'] = df['percentile75_ran'] - df['mean_ran']
df['lower_error_ran'] = df['mean_ran'] - df['percentile25_ran']

x = df['x1 pytorch_N']
y1 = df['mean_badge']
y2 = df['mean_unc']
y3 = df['mean_ran']

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(x, y1, color='r', label='BADGE')
plt.plot(x, y2, color='b', label='UNCERTAINTY')
plt.plot(x, y3, color='g', label='RANDOM')

# Fill between the 25th and 75th percentiles
plt.fill_between(x, y1 - df['lower_error_badge'], y1 + df['upper_error_badge'], color='red', alpha=0.5)
plt.fill_between(x, y2 - df['lower_error_unc'], y2 + df['upper_error_unc'], color='lightblue', alpha=0.5)
plt.fill_between(x, y3 - df['lower_error_ran'], y3 + df['upper_error_ran'], color='lightgreen', alpha=0.5)

plt.xlabel('# of data queried (galaxy image)')
plt.ylabel('Accuracy Score')
plt.title('Performance of one-layer MLP with 200 sample selected each iter')
plt.legend()
plt.grid(True)
plt.show()
