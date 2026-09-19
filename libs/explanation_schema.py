"""Domain types for explanation only; never change benchmark preprocessing.

Credit-g's legacy OpenML metadata calls these coded variables numeric.
The corrected documentation identifies ordinal/binary categories:
https://archive.ics.uci.edu/dataset/573/south+german+credit+update
Keep original codes rather than inventing category interval labels.
"""

CREDIT_G_CODED_CATEGORIES = {
    "installment_commitment": (1, 2, 3, 4),
    "residence_since": (1, 2, 3, 4),
    "existing_credits": (1, 2, 3, 4),
    "num_dependents": (1, 2),
}


def explanation_categories(dataset_id):
    """Explicit schema overrides, never a heuristic based on integer values."""
    return CREDIT_G_CODED_CATEGORIES.copy() if int(dataset_id) == 31 else {}
