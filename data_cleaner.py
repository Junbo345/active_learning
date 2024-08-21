import pandas as pd

def label_highest_prob(row):
    """
    Convert a row of probabilities to a one-hot encoded vector where the highest probability is set to 1.
    """
    max_value_index = row.argmax()  # Find the index of the maximum value in the row
    new_row = [0] * len(row)  # Set all values to 0
    new_row[max_value_index] = 1  # Set the index of the maximum value to 1
    return new_row


def loaddata(feature_cols, label_cols) -> pd.DataFrame:
    """
    Load the galaxy dataset, preprocess it, and return a DataFrame with features and one-hot encoded labels.
    """
    expected_csv_loc = 'galaxy.csv'  # Expect the CSV file to be in the current folder
    df = pd.read_csv(expected_csv_loc)
    df['other'] = 1 - (df['s1_lrg_fraction'] + df['s1_spiral_fraction'])

    # Load the data and select specified columns
    columns_to_select = feature_cols + label_cols
    df_s = df[columns_to_select]

    # Apply the label_highest_prob function to each row and convert to list of integers
    label_columns_expanded = df_s[label_cols].apply(label_highest_prob, axis=1, result_type='expand')

    # Ensure the expanded label columns have the same names as the original label columns
    label_columns_expanded.columns = label_cols

    # Concatenate the feature columns with the expanded label columns
    df_final = pd.concat([df_s[feature_cols], label_columns_expanded], axis=1)

    return df_final

