from __future__ import annotations


def four_way_time_split(df, train_fraction=0.60, selection_fraction=0.15,
                        conformal_fraction=0.10, test_fraction=0.15):
    total = train_fraction + selection_fraction + conformal_fraction + test_fraction
    if abs(total - 1.0) > 1e-8:
        raise ValueError("Split fractions must sum to 1.0")
    n = len(df)
    a = int(n * train_fraction)
    b = a + int(n * selection_fraction)
    c = b + int(n * conformal_fraction)
    if not (0 < a < b < c < n):
        raise ValueError("Not enough rows for requested split")
    return (
        df.iloc[:a].copy(),
        df.iloc[a:b].copy(),
        df.iloc[b:c].copy(),
        df.iloc[c:].copy(),
    )


def chronological_split(df, train_fraction=0.60, calibration_fraction=0.15):
    # Backward-compatible 3-way split used by the synthetic demo.
    n = len(df)
    train_end = int(n * train_fraction)
    cal_end = int(n * (train_fraction + calibration_fraction))
    if not (0 < train_end < cal_end < n):
        raise ValueError("Invalid chronological split fractions.")
    return df.iloc[:train_end].copy(), df.iloc[train_end:cal_end].copy(), df.iloc[cal_end:].copy()
