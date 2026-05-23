"""RL training utilities: hyperparameter narrowing, fee curriculum."""


def narrow(val: float, lo: float, hi: float, pct: float = 0.50) -> tuple[float, float]:
    """Narrow a float range [lo, hi] to ±pct around val, clamped to original bounds.

    Used in Optuna Phase 2 (CMA-ES) to define refined search ranges around
    the best parameters found in Phase 1 (TPE).

    Args:
        val: centre value (best param from Phase 1).
        lo:  original lower bound.
        hi:  original upper bound.
        pct: fraction of val to use as half-width (default ±50%).

    Returns:
        (new_lo, new_hi) clamped to [lo, hi].
    """
    delta = abs(val) * pct
    new_lo = max(lo, val - delta)
    new_hi = min(hi, val + delta)
    # Guard: val outside [lo, hi] can produce an inverted interval after clamping.
    # Ensure lo <= hi by returning the clamped endpoints in sorted order.
    return (new_lo, new_hi) if new_lo <= new_hi else (new_hi, new_lo)


def narrow_int(val: int, lo: int, hi: int, pct: float = 0.50) -> tuple[int, int]:
    """Narrow an int range [lo, hi] to ±pct around val, minimum delta of 1.

    Args:
        val: centre value (best param from Phase 1).
        lo:  original lower bound.
        hi:  original upper bound.
        pct: fraction of val to use as half-width (default ±50%).

    Returns:
        (new_lo, new_hi) as integers, clamped to [lo, hi].
    """
    delta = max(1, int(round(abs(val) * pct)))
    new_lo = max(lo, val - delta)
    new_hi = min(hi, val + delta)
    return (new_lo, new_hi) if new_lo <= new_hi else (new_hi, new_lo)


def fee_curriculum_factor(
    timestep: int,
    total_timesteps: int,
    warmup_frac: float = 0.10,
    ramp_end_frac: float = 0.50,
) -> float:
    """Fee curriculum factor: 0.0 during warmup, linear ramp, then 1.0.

    Three phases proportional to total_timesteps:
    - Phase 1 (0 → warmup):          factor = 0.0  (exploration without costs)
    - Phase 2 (warmup → ramp_end):   factor linearly 0 → 1
    - Phase 3 (ramp_end → end):      factor = 1.0  (full real fees)

    Args:
        timestep:        current training timestep.
        total_timesteps: total timesteps for this training run.
        warmup_frac:     fraction of total_timesteps with zero fees (default 0.10).
        ramp_end_frac:   fraction at which fees reach 1.0 (default 0.50).

    Returns:
        Multiplicative fee factor in [0.0, 1.0].
    """
    warmup   = warmup_frac   * total_timesteps
    ramp_end = ramp_end_frac * total_timesteps
    if timestep < warmup:
        return 0.0
    if timestep < ramp_end:
        return (timestep - warmup) / (ramp_end - warmup)
    return 1.0
