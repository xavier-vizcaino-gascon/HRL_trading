"""RL evaluation metrics: Sharpe ratio, max drawdown."""
import numpy as np


def compute_sharpe(equities: list, steps_per_year: int = 365 * 288) -> float:
    """Ràtio de Sharpe anualitzat a partir d'una corba d'equitat.

    Args:
        equities:       llista de valors d'equitat (escalar per pas).
        steps_per_year: nombre de passos per any (288 barres 5m × 365 dies).

    Returns:
        Sharpe anualitzat. Retorna 0.0 si la desv. estàndard és quasi-zero.
    """
    eq  = np.array(equities, dtype=np.float64)
    if len(eq) < 2:
        return 0.0
    ret = np.diff(eq) / (eq[:-1] + 1e-10)
    std = ret.std()
    if std < 1e-10:
        return 0.0
    return float(ret.mean() / std * np.sqrt(steps_per_year))


def compute_max_drawdown(equities: list) -> float:
    """Màxim drawdown en percentatge sobre la corba d'equitat.

    Args:
        equities: llista de valors d'equitat.

    Returns:
        Màxim drawdown (%) com a valor positiu.
    """
    eq  = np.array(equities, dtype=np.float64)
    run = np.maximum.accumulate(eq)
    dd  = (eq - run) / (run + 1e-10)
    return float(abs(dd.min()) * 100)
