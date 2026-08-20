import numpy as np
from sklearn.metrics import precision_recall_fscore_support

def detection_metrics(y_true, y_pred):
    p,r,f1,_ = precision_recall_fscore_support(
        np.asarray(y_true).astype(int),
        np.asarray(y_pred).astype(int),
        average="binary",
        zero_division=0,
    )
    return {"precision":float(p), "recall":float(r), "f1":float(f1)}

def carbon_debt_error(true_debt, pred_debt):
    true_debt = float(true_debt)
    pred_debt = float(pred_debt)
    return {
        "absolute_error_kg": abs(pred_debt-true_debt),
        "relative_error": abs(pred_debt-true_debt)/(abs(true_debt)+1e-9),
    }
