__all__ = [
    "calculate_entropy",
    "calculate_pointwise_marginal",
    "calculate_joint_entropy",
    "calculate_conditional_entropy",
    "calculate_conditional_entropyWithMarginal",
    "calculate_pointwise_conditional_entropy",
    "calculateMutualInformation",
    "calculateJointDistribution",
    "conditionalPrivacyScore",
    "conditionalPrivacyScoreMIL",
    "calculate_pointwise_mutual_information",
    "maximum_information_leakage",
    "maximum_information_leakageGPT",
    "combinedIPTR",
    "calculate_generalised_entropy",
    "filteredDfbasedOnValues",
    "calculate_generalised_pointwise_marginal",
    "calculate_pointwise_generalised_conditional_entropy",
    "calculate_entropy_l_diversityGPT",
]

from loguru import logger
import typing as ty
import pandas as pd
import random
import math
import numpy as np


def calculate_entropy(x: pd.Series) -> float:
    """Entropy of column (single feature X).

    Args:
        x (pd.Series):
            Single column.

    Returns:
        float: Entropy value
    """
    count = x.value_counts()
    probabilities = count / len(x)
    entropy = float(-sum(probabilities * np.log2(probabilities)))
    return entropy


# P(Y=y)
def calculate_pointwise_marginal(x: pd.Series, value: float) -> float:
    """Proability of seeing Y=y (frequence).

    Args:
        x (pd.Series):
            Column.
        value (float):
            Value.

    Returns:
        float: _description_
    """
    # Count the number of occurrences of y_value in Y
    count_y_value = (x == value).sum()
    # Total number of records
    total_records = len(x)
    # Calculate the probability
    probability_y: float = float(count_y_value / total_records)
    return probability_y


def calculate_joint_entropy(column1: pd.Series, column2: pd.Series) -> float:
    """Joint entropy calculation.

    Args:
        column1 (pd.Series): _description_
        column2 (pd.Series): _description_

    Returns:
        float: Joint entropy.
    """
    probabilities = calculateJointDistribution(column1, column2)
    joint_entropy = float(-sum(probabilities * np.log2(probabilities)))
    return joint_entropy


def calculate_conditional_entropy(X: pd.Series, Y: pd.Series) -> float:
    """H(X|Y) = H(X,Y) - H(Y)

    Args:
        X (pd.Series): _description_
        Y (pd.Series): _description_

    Returns:
        float: _description_
    """
    joint_entropy_XY = calculate_joint_entropy(X, Y)
    entropy_Y = calculate_entropy(Y)
    return joint_entropy_XY - entropy_Y


# H(X|Y)= - sum (p(x,y)x log(p(x|y)))
def calculate_conditional_entropyWithMarginal(column1: pd.Series, column2: pd.Series) -> float:
    """H(X|Y)= - sum (p(x,y)x log(p(x|y)))

    Args:
        column1 (pd.Series): _description_
        column2 (pd.Series): _description_

    Returns:
        float: _description_
    """
    # Calculate joint distribution p(x, y)
    joint_prob = calculateJointDistribution(column1, column2)
    # Calculate marginal distribution p(y)
    marginal_prob_Y = column2.value_counts() / len(column2)
    # Calculate conditional entropy
    conditional_entropy = 0
    for joint_event, p_xy in joint_prob.items():
        x, y = joint_event.split("_")  # type: ignore
        p_y = marginal_prob_Y[y]
        p_x_given_y = p_xy / p_y if p_y > 0 else 0
        conditional_entropy -= p_xy * math.log(p_x_given_y, 2) if p_x_given_y > 0 else 0
    return conditional_entropy


# H(X|Y=y_value)
def calculate_pointwise_conditional_entropy(
    X: pd.Series,
    Y: pd.Series,
    value: float,
) -> float:
    """H(X|Y=y)

    Args:
        X (pd.Series):
            First column.
        Y (pd.Series):
            Second column.
        value (float):
            Value of `Y` to condition for, to estimate `H(X | Y=value)`.

    Returns:
        float: _description_
    """
    # Filtering Y for the specific value
    idx = Y == value
    # Calculate conditional entropy of X given this filtered version of Y
    H_X_Y = calculate_entropy(X[idx])
    return H_X_Y


# I(X;Y) = h(X) - h(X|Y) = H(X) + H(Y)  -H(X,Y)
def calculateMutualInformation(column1: pd.Series, column2: pd.Series) -> float:
    """I(X;Y) = h(X) - h(X|Y) = H(X) + H(Y)  -H(X,Y)

    Args:
        column1 (pd.Series): _description_
        column2 (pd.Series): _description_

    Returns:
        float: _description_
    """
    return calculate_entropy(column1) + calculate_entropy(column2) - calculate_joint_entropy(column1, column2)


def calculateJointDistribution(column1: pd.Series, column2: pd.Series) -> pd.Series:
    """_summary_

    Args:
        column1 (pd.Series): _description_
        column2 (pd.Series): _description_

    Returns:
        float: _description_
    """
    joint_data = column1.astype(str) + "_" + column2.astype(str)
    joint_count = joint_data.value_counts()
    joint_prob = joint_count / len(joint_data)
    return joint_prob


# score = 1 - 2 exp -I(X;Y) this is a normalised score of the mutula information
def conditionalPrivacyScore(column1: pd.Series, column2: pd.Series) -> float:
    """score = 1 - 2 exp -I(X;Y) this is a normalised score of the mutula information

    Args:
        column1 (pd.Series): _description_
        column2 (pd.Series): _description_

    Returns:
        float: _description_
    """
    return 1 - math.pow(2, -calculateMutualInformation(column1, column2))


# score = 1 - 2 exp -MIL(X;Y) this is a normalised score of the MIL
def conditionalPrivacyScoreMIL(column1: pd.Series, column2: pd.Series) -> float:
    """score = 1 - 2 exp -MIL(X;Y) this is a normalised score of the MIL

    Args:
        column1 (pd.Series): _description_
        column2 (pd.Series): _description_

    Returns:
        float: _description_
    """
    return 1 - math.pow(2, -maximum_information_leakageGPT(column1, column2))


# Maximum informaiton leak (MIL) the max amount of information about a sensitive attribute that can be learned  by observing a single instance of Y.
def calculate_pointwise_mutual_information(column1: pd.Series, column2: pd.Series, value: float) -> float:
    """Maximum informaiton leak (MIL) the max amount of information about a sensitive attribute that can be learned  by observing a single instance of Y.

    Args:
        column1 (pd.Series): _description_
        column2 (pd.Series): _description_
        value (float): _description_

    Returns:
        float: _description_
    """
    # Create a binary columns for Y where each entry is True if Y equals the specified value
    column2_binary = column2.apply(lambda y: y == value)
    # you're essentially trying to measure how much information the event "Column2 equals y" provides about Column1.
    #  This is different from measuring the mutual information between the entire distributions of Column1 and Column2.
    #  Instead, you're focusing on the information provided by a specific event (Column2 being equal to y) about the entire distribution of Column1.
    # Calculate mutual information between column1 and this binary event
    return calculateMutualInformation(column1, column2_binary)


# This metric essentially captures the most an adversary can learn about the input by observing the output in the worst-case scenario.
# It's a powerful concept in scenarios where you need to evaluate how much sensitive information could potentially be inferred or leaked through observable data.
def maximum_information_leakage(column1: pd.Series, column2: pd.Series) -> float:
    """This metric essentially captures the most an adversary can learn about the input by observing the output in the worst-case scenario.
    It's a powerful concept in scenarios where you need to evaluate how much sensitive information could potentially be inferred or leaked through observable data.

        Args:
            column1 (pd.Series): _description_
            column2 (pd.Series): _description_

        Returns:
            float: _description_
    """
    unique_values_column2 = column2.unique()
    max_info_leakage = 0.0

    for val in unique_values_column2:
        mutual_info = calculate_pointwise_mutual_information(column1, column2, val)
        max_info_leakage = max(max_info_leakage, mutual_info)

    return max_info_leakage


# MIL Generated by GPT but given result closer to the paper.
def maximum_information_leakageGPT(X: pd.Series, Y: pd.Series) -> float:
    """MIL Generated by GPT but given result closer to the paper.

    Args:
        X (pd.Series): _description_
        Y (pd.Series): _description_

    Returns:
        float: _description_
    """
    unique_values_Y = Y.unique()
    max_info_leakage = 0.0

    for value in unique_values_Y:
        mutual_info = calculate_pointwise_mutual_information(X, Y, value)
        max_info_leakage = max(max_info_leakage, mutual_info)

    return max_info_leakage


# dataset is the table
# X_columnName the string name of the sensitive attribute
# target_Y_column_names the list of column names for quasi identifiers ['attribute1', 'attribute2', ...]


def combinedIPTR(dataset: pd.DataFrame, x_name: str) -> float:
    """_summary_

    Args:
        dataset (pd.DataFrame): the table
        x_name (str): name of the sensitive attribute

    Returns:
        float: _description_
    """
    # Create the X column as data frame
    x = dataset[x_name]
    # Calculate the entropy of the target attribute
    hx = calculate_entropy(pd.Series(x.unique()))
    return hx


# H(Y1,Y2,..Yn)
def calculate_generalised_entropy(columns: pd.Series) -> float:
    """H(Y1,Y2,..Yn)

    Args:
        columns (pd.Series): _description_

    Returns:
        float: _description_
    """
    probabilities = columns.value_counts(normalize=True)
    entropy = float(-np.sum(probabilities * np.log2(probabilities)))
    return entropy


def filteredDfbasedOnValues(
    df: pd.DataFrame,
    conditional_cols: ty.List[str],
    conditional_values: ty.List[float],
) -> pd.DataFrame:
    """Filter DataFrame based on conditional values."""
    for col, val in zip(conditional_cols, conditional_values):
        df = df[df[col] == val]
    return df


# Equation 10 of the paper
# p(Y1=y1, Y2 =y2, ....Yn= yn)= probablity of occurence of y1, ...yn
def calculate_generalised_pointwise_marginal(
    df: pd.DataFrame,
    conditional_cols: ty.List[str],
    conditional_values: ty.List[ty.Any],
) -> float:
    """Equation 10 of the paper.
    p(Y1=y1, Y2 =y2, ....Yn= yn)= probablity of occurence of y1, ...yn.

    Args:
        df (pd.DataFrame): _description_
        conditional_cols (ty.List[str]): _description_
        conditional_values (ty.List[float]): _description_

    Returns:
        float: _description_
    """
    # filtered combined column filter
    fitlered_df = filteredDfbasedOnValues(df, conditional_cols, conditional_values)
    # print(" DF:", df.to_string())
    # print("filtered DF:", fitlered_df.to_string())
    if len(fitlered_df) == 0:
        return 0
    # Count the number of occurrences of y1, y2,...yn
    count_yi_value = len(fitlered_df)
    # print("counnt filtered DF:", count_yi_value)

    # Total number of records
    # Question: do we need to drop duplicate to calculate the total number of row
    # in other word if p(Y=y1, Y2= y2, ...Yn=yn) = count (tuple where Y=y1, Y2= y2, ...Yn=yn)/total count, is total count the number of rows or the number of unique row values ?
    # Drop duplicate rows based on the conditional columns. I choose to remove the drop duplicate otherwie the number of rows fitered can be > total number of tuple without duplcated => proba >1
    # unique_combinations = df[conditional_cols].drop_duplicates()
    unique_combinations = df[conditional_cols]

    # Count the number of unique combinations
    num_unique_combinations = len(unique_combinations)

    # Calculate the probability
    probability_yi = count_yi_value / num_unique_combinations

    return probability_yi


def calculate_pointwise_generalised_conditional_entropy(
    df: pd.DataFrame,
    target_col: str,
    conditional_cols: ty.List[str],
    conditional_values: ty.List[ty.Any],
) -> float:
    """H(X|Y1=y1,Y2=y2,..Yn=yn)
    conditional_cols = ['Y1', 'Y2']  # The names of the conditional columns
    conditional_values = ['y1_value', 'y2_value']  # The target values for each conditional column
    conditional_entropy = calculate_conditional_entropy(df, 'X', conditional_cols, conditional_values)
    For example, if conditional_cols = ['Y1', 'Y2'] and conditional_values = ['y1_value', 'y2_value'],
    the DataFrame df will first be filtered to include only those rows where Y1 is 'y1_value'.
    Then, this already filtered DataFrame is further filtered to include only those rows where Y2 is 'y2_value'.
    The end result is a DataFrame that includes only the rows where each column in conditional_cols matches its corresponding value in conditional_values. This filtered DataFrame is used for subsequent calculations, such as computing the conditional entropy.
    """
    assert len(conditional_cols) == len(conditional_values), "Columns and values length must match"
    # filter df based on values
    df = filteredDfbasedOnValues(df, conditional_cols, conditional_values)
    # Check if the filtered DataFrame is not empty
    if len(df) == 0:
        return 0.0
    # Calculate and return the entropy of the target column in the filtered DataFrame
    return calculate_generalised_entropy(df[target_col])


def calculate_entropy_l_diversityGPT(
    table: pd.DataFrame,
    quasi_identifier_col: str,
    sensitive_col: str,
    l: float,
) -> bool:
    """_summary_

    Args:
        table (pd.DataFrame): _description_
        quasi_identifier_col (str): _description_
        sensitive_col (str): _description_
        l (float): _description_

    Returns:
        bool: _description_
    """
    # Group by the quasi-identifier
    groups = table.groupby(quasi_identifier_col)

    # Check the entropy l-diversity condition for each group
    for group_name, group in groups:
        # Calculate the distribution of the sensitive attribute within the group
        distribution = group[sensitive_col].value_counts() / len(group)

        # Calculate the entropy of this distribution
        entropy = -np.sum(distribution * np.log2(distribution))

        # Check if entropy is at least log(l)
        if entropy < math.log(l, 2):
            return False  # Not entropy l-diverse

    return True  # Entropy l-diverse for all groups
