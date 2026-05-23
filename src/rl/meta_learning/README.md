# Meta-Learning Module

Implementació de meta-aprenentatge adaptatiu per agents RL.

## Mòduls

### `adaptive_reptile.py`
Sistema Reptile adaptatiu amb convergència automàtica:
- `InnerConvergenceCriteria`: Criteris de convergència per inner loops
- `OuterUpdateTrigger`: Quan fer outer updates
- `MetaConvergenceCriteria`: Quan parar meta-training
- `AdaptiveReptile`: Orquestrador principal Reptile

### `adaptive_fomaml.py`
Sistema FOMAML adaptatiu (First-Order MAML):
- `AdaptiveFOMAML`: Orquestrador principal FOMAML
- Reutilitza criteris de `adaptive_reptile.py`

### `diagnostics.py`
Diagnòstics detallats per inner loops:
- `measure_inner_convergence()`: Mesura convergència d'un inner loop
- `plot_inner_convergence()`: Visualitza trajectòries de convergència

## Ús

```python
from src.rl.meta_learning import (
    AdaptiveReptile,
    InnerConvergenceCriteria,
    OuterUpdateTrigger,
    MetaConvergenceCriteria,
)

# Defineix criteris
inner_criteria = InnerConvergenceCriteria(
    min_steps=5_000, max_steps=30_000, target_sharpe=0.8
)
outer_trigger = OuterUpdateTrigger(batch_size=5)
meta_criteria = MetaConvergenceCriteria(max_meta_iters=500)

# Crea sistema
reptile = AdaptiveReptile(
    inner_criteria, outer_trigger, meta_criteria,
    device, ppo_config, reptile_eps=0.1
)

# Entrena (para automàticament quan convergeix!)
result = reptile.meta_train(
    base_policy, task_sampler, env_factory, val_df
)
```

## Documentació completa

Consulta els notebooks `06*_ppo_*_adaptive.ipynb` a:
- `notebooks/modul3_dense/` (dense reward)
- `notebooks/modul3_sparse/` (sparse reward)

I la documentació de guies:
- `INNER_CONVERGENCE_GUIDE.md` - Interpretació de convergència
- `REPTILE_VS_FOMAML.md` - Comparativa algorismes
- `README_ADAPTIVE_META_LEARNING.md` - Guia d'ús completa
