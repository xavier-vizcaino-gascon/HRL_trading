"""
Meta-learning algorithms for RL agents.

This module implements adaptive meta-learning systems:
- Reptile: First-order meta-learning via parameter interpolation
- FOMAML: First-order MAML via gradient descent on query loss
"""

from src.rl.meta_learning.adaptive_reptile import (
    AdaptiveReptile,
    InnerConvergenceCriteria,
    OuterUpdateTrigger,
    MetaConvergenceCriteria,
)

from src.rl.meta_learning.adaptive_fomaml import AdaptiveFOMAML

from src.rl.meta_learning.diagnostics import (
    measure_inner_convergence,
    plot_inner_convergence,
)

__all__ = [
    # Reptile
    'AdaptiveReptile',
    'InnerConvergenceCriteria',
    'OuterUpdateTrigger',
    'MetaConvergenceCriteria',

    # FOMAML
    'AdaptiveFOMAML',

    # Diagnostics
    'measure_inner_convergence',
    'plot_inner_convergence',
]
