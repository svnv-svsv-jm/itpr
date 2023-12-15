__all__ = ["TABLE"]

import pandas as pd


N = 8
ids = [f"{i}" for i in range(N)]
data = {
    "id": ids,
    # Age
    "age_1": [30, 62, 37, 21, 19, 47, 71, 73],
    "age_2": [30] * N,
    "age_3": [30, 30, 30, 47, 30, 30, 30, 30],
    "age_4": [30, 30, 30, 47, 30, 47, 30, 30],
    "age_5": [30, 30, 30, 30, 47, 47, 47, 47],
    # Disease
    "disease_1": ["diabetes", "epilepsy", "asthma", "allergies", "depession", "hiv", "heart", "cancer"],
    "disease_2": ["diabetes", "diabetes", "epilepsy", "depession", "hiv", "heart", "cancer", "allergies"],
    "disease_3": ["diabetes", "diabetes", "diabetes", "diabetes", "diabetes", "hiv", "asthma", "allergies"],
}
TABLE = pd.DataFrame(data)
