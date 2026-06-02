"""
src/evaluate.py
Métriques partagées — à utiliser par tous les membres.
PAS d'accuracy — F1-Macro, AUPRC, MCC uniquement.

Usage:
    from src.evaluate import evaluate, print_report
"""

from sklearn.metrics import (
    f1_score, average_precision_score, matthews_corrcoef,
    classification_report, confusion_matrix
)
import pandas as pd


def evaluate(y_true, y_pred, y_proba, label='', verbose=True):
    """Calcule et retourne les métriques standard du projet."""
    f1    = f1_score(y_true, y_pred, average='macro')
    auprc = average_precision_score(y_true, y_proba)
    mcc   = matthews_corrcoef(y_true, y_pred)
    f1_fraud = f1_score(y_true, y_pred, average=None)[1]
    recall   = (y_pred[y_true == 1] == 1).mean()
    precision = (y_true[y_pred == 1] == 1).mean() if (y_pred==1).sum()>0 else 0

    if verbose:
        print(f"
{'='*45}")
        print(f"  {label}")
        print(f"{'='*45}")
        print(f"  F1-Macro  : {f1:.4f}")
        print(f"  AUPRC     : {auprc:.4f}")
        print(f"  MCC       : {mcc:.4f}")
        print(f"  F1-Fraude : {f1_fraud:.4f}")
        print(f"  Rappel    : {recall:.4f} ({int(recall*y_true.sum())}/{y_true.sum()} fraudes)")
        print(f"  Précision : {precision:.4f}")

    return {
        'Modele'    : label,
        'F1-Macro'  : round(f1, 4),
        'AUPRC'     : round(auprc, 4),
        'MCC'       : round(mcc, 4),
        'F1-Fraude' : round(f1_fraud, 4),
        'Rappel'    : round(recall, 4),
        'Precision' : round(precision, 4),
    }


def compare_models(results_list):
    """Affiche un tableau comparatif propre de plusieurs modèles."""
    df = pd.DataFrame(results_list).set_index('Modele')
    print("
" + "="*60)
    print("         COMPARAISON DES MODÈLES")
    print("="*60)
    print(df.to_string())
    return df