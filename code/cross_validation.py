"""
cross_validation.py

Subject-independent Group K-Fold cross-validation
for Parkinson's Disease Detection.
"""

from sklearn.model_selection import GroupKFold


def verify_groupkfold(X_fused, y, subjects, n_splits=5):
    """
    Verify subject-independent Group K-Fold splitting.

    Parameters
    ----------
    X_fused : ndarray
        Feature matrix.

    y : ndarray
        Labels.

    subjects : ndarray
        Subject IDs.

    n_splits : int
        Number of folds.

    Returns
    -------
    GroupKFold object
    """

    gkf = GroupKFold(n_splits=n_splits)

    print("=" * 70)
    print("GROUP K-FOLD VERIFICATION")
    print("=" * 70)

    for fold, (train_idx, test_idx) in enumerate(
            gkf.split(X_fused, y, groups=subjects), 1):

        train_subjects = set(subjects.iloc[train_idx])
        test_subjects = set(subjects.iloc[test_idx])

        overlap = train_subjects.intersection(test_subjects)

        print(f"\nFold {fold}")
        print("-" * 50)

        print("Train Samples :", len(train_idx))
        print("Test Samples  :", len(test_idx))

        print("Train Subjects:", len(train_subjects))
        print("Test Subjects :", len(test_subjects))

        print("Overlap Subjects :", len(overlap))

        if len(overlap) == 0:
            print("✓ Subject Independent")
        else:
            print("✗ Subject Leakage Found")

    return gkf
