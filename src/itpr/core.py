__all__ = ["calculate_ITPR"]

from loguru import logger
import typing as ty
import random
import pandas as pd

from .utils import (
    calculate_entropy,
    calculate_pointwise_marginal,
    calculate_pointwise_conditional_entropy,
    calculate_joint_entropy,
    calculate_conditional_entropy,
    calculate_conditional_entropyWithMarginal,
    calculateMutualInformation,
    maximum_information_leakage,
    conditionalPrivacyScore,
    maximum_information_leakageGPT,
    conditionalPrivacyScoreMIL,
    calculate_pointwise_generalised_conditional_entropy,
    calculate_generalised_pointwise_marginal,
)


def calculate_ITPR(
    X: pd.Series,
    Y: pd.Series,
    tol: float = 1e-12,
) -> float:
    """ITPR.

    Args:
        X (pd.Series):
            First column.
        Y (pd.Series):
            Second column.
        tol (float, optional):
            Tolerance to avoid divide-by-zero errors on division operation.

    Returns:
        float:
            ITPR value.
    """
    y_values = Y.unique()
    omega_Y_size = len(y_values)
    logger.trace(f"({omega_Y_size}) y_values={y_values}")
    # H(X))
    H_X = calculate_entropy(X)
    logger.trace(f"H(X)={H_X}")
    # ITPR(Y) over X
    max_itpr_value = 0.0
    for y_value in y_values:
        probability_y = calculate_pointwise_marginal(Y, y_value)
        logger.trace(f"probability_y={probability_y}")
        H_X_Y = calculate_pointwise_conditional_entropy(X, Y, y_value)
        logger.trace(f"H(X|Y)={H_X_Y}")
        weighted_conditional_entropy = omega_Y_size * probability_y * H_X_Y
        logger.trace(f"weighted_conditional_entropy={weighted_conditional_entropy}")
        itpr_value = 1 - (weighted_conditional_entropy / (H_X + tol))
        max_itpr_value = max(max_itpr_value, itpr_value)

    return max_itpr_value


def calculate_generalised_ITPR(
    dataset: pd.DataFrame,
    target_col: str,
    conditional_cols: ty.List[str],
) -> float:
    """Generalised ITPR(X, Y1, Y2, ..., YN).

    Args:
        dataset (pd.DataFrame):
            Input table.
        target_col (str):
            Name of the target column. `X`
        conditional_cols (ty.List[str]):
            List of column names, for the conditional columns `Y1, Y2, ..., YN`.

    Returns:
        float: Value of the ITPR.
    """
    # Create DataFrame with the specified attributes
    X = dataset[target_col]
    # Get unique tuples from the DataFrame
    X = X.drop_duplicates()
    # Calculate the entropy of the target attribute
    H_X = calculate_entropy(X)
    # build the Tau data set
    Y_dataset = dataset[conditional_cols].drop_duplicates()
    # Iterate over Tau to get Y1=y1, ....Yn=yn
    max_itpr_value = 0.0
    for tup in Y_dataset.itertuples(index=False):
        # in this case the Y1=y1, ....Yn=yn is the current tuple in Tau data set
        weighted_cond_entropy = calculate_generalised_pointwise_marginal(
            dataset, conditional_cols, tup
        ) * calculate_pointwise_generalised_conditional_entropy(dataset, target_col, conditional_cols, tup)
        itpr_value = 1 - (len(Y_dataset) * weighted_cond_entropy / H_X)
        max_itpr_value = max(max_itpr_value, itpr_value)
    # Return
    return max_itpr_value


if __name__ == "__main__":
    # Generate synthetic data
    k = 100
    names = [f"Person_{i}" for i in range(k)]
    phone_numbers = [f"123-456-78{i:02d}" for i in range(k)]
    addresses = [f"Address_{i}" for i in range(k)]
    birthdates = [f"199{random.randint(0, 9)}-01-01" for _ in range(k)]
    customer_types = ["VIP", "normal", "low"]

    data = {
        "name": names,
        "phone_number": phone_numbers,
        "address": addresses,
        "birthdate": birthdates,
        "type_of_customer": random.choices(customer_types, k=k),
    }

    table = pd.DataFrame(data)
    print("The generated table:")
    print(table.to_string())

    # paper Table
    age = ["30", "30", "30", "47", "30", "47", "30", "30"]
    disease = ["Diabetes", "Diabetes", "Epilepsy", "Depression", "HIV", "Heart", "Cancer", "Allergies"]
    assert len(age) == len(disease)
    k_paper = len(age)
    ids = [f"{i}" for i in range(k_paper)]
    patientType = ["heavy", "normal"]

    paperData = {
        "id": ids,
        "age": age,
        "disease": disease,
        "type": random.choices(patientType, k=k_paper),
    }
    paperTable = pd.DataFrame(paperData)

    # Choose columns for entropy calculations
    column_X = table["birthdate"]
    column_Y = table["type_of_customer"]

    # Calculate entropies
    entropy_X = calculate_entropy(column_X)
    entropy_Y = calculate_entropy(column_Y)
    joint_entropy_XY = calculate_joint_entropy(column_X, column_Y)
    conditional_entropy_X_given_Y = calculate_conditional_entropy(column_X, column_Y)
    conditional_entropy_X_given_Y_w_Marginal = calculate_conditional_entropyWithMarginal(column_X, column_Y)

    print("Entropy of X (birthdate):", entropy_X)
    print("Entropy of Y (type_of_customer):", entropy_Y)
    print("Joint Entropy of X and Y:", joint_entropy_XY)
    print("Conditional Entropy of X given Y:", conditional_entropy_X_given_Y)
    print(f"Conditional Entropy caclculated by marginal P(y): {conditional_entropy_X_given_Y_w_Marginal}")
    print("Mutual information", calculateMutualInformation(column_X, column_Y))
    print("Maximum information leakage", maximum_information_leakage(column_X, column_Y))

    # Privacy score
    print("conditional privacy score:", conditionalPrivacyScore(column_X, column_Y))
    print(f"conditional privacy score (MIL) worst case: {conditionalPrivacyScoreMIL(column_X, column_Y)}")

    # On the paper table
    print("*****************paper table **************")
    print(paperTable.to_string())
    # Choose columns for entropy calculations
    column_X = paperTable["disease"]
    column_Y = paperTable["age"]
    print("Mutual information", calculateMutualInformation(column_X, column_Y))
    print(
        "Maximum information leakage GPT",
        maximum_information_leakageGPT(column_X, column_Y),
    )
    print("Maximum information leakage", maximum_information_leakage(column_X, column_Y))

    # Privacy score
    print("conditional privacy score:", conditionalPrivacyScore(column_X, column_Y))
    print(
        "conditional privacy score (MIL) worst case:",
        conditionalPrivacyScoreMIL(column_X, column_Y),
    )
    print("IPTR: ", calculate_ITPR(column_X, column_Y))

    print(
        "calculate point wise entropy generalised ",
        calculate_pointwise_generalised_conditional_entropy(
            paperTable, "disease", ["age", "type"], ["30", "normal"]
        ),
    )
    print(
        "Calculate point wise marginal",
        calculate_generalised_pointwise_marginal(paperTable, ["age", "type"], ["30", "normal"]),
    )
    print(
        "Generalised IPTR: ",
        calculate_generalised_ITPR(paperTable, "disease", ["age", "type"]),
    )
