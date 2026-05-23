"""Synthetic data generation for HRL pre-training.

Generates Polars DataFrames with realistic Norm_* features + OHLC from sinusoidal prices.
Compatible with CryptoMarketEnvConsolidatedReward.

Usage:
    from src.rl.training.synthetic_data import make_synthetic_df

    feature_cols = dm.features  # list of 65 Norm_* column names
    df_synth = make_synthetic_df(
        n_steps=10_000,
        phase=1,
        feature_cols=feature_cols,
        seed=42,
    )

    env = CryptoMarketEnvConsolidatedReward(df_synth, feature_cols)
"""

from __future__ import annotations

import numpy as np
import polars as pl


def generate_synthetic_prices(
    n_steps: int,
    phase: int,
    base_price: float = 50_000.0,
    amplitude_pct: float = 0.10,
    base_freq: float = 0.02,
    freq_sweep_range: tuple[float, float] = (0.005, 0.05),
    fractal_freqs: list[float] | None = None,
    fractal_amplitudes: list[float] | None = None,
    seed: int = 42,
) -> np.ndarray:
    """Generate synthetic close prices from sinusoidal patterns.

    Args:
        n_steps: Number of timesteps to generate
        phase: 1 (fixed freq), 2 (chirp/sweep), or 3 (fractal/multi-freq)
        base_price: Base price level (e.g., 50000 for BTC)
        amplitude_pct: Oscillation amplitude as fraction of base_price
        base_freq: Base frequency in radians/step (phase 1)
        freq_sweep_range: (f_min, f_max) for phase 2 chirp
        fractal_freqs: List of frequencies for phase 3 (e.g., [0.005, 0.02, 0.08, 0.2])
        fractal_amplitudes: List of amplitudes for phase 3 (e.g., [0.06, 0.03, 0.015, 0.005])
        seed: Random seed for reproducibility

    Returns:
        1D array of close prices, shape (n_steps,)

    Examples:
        Phase 1: close[t] = base * (1 + A * sin(f * t))
        Phase 2: close[t] = base * (1 + A * sin(φ(t))) where φ(t) sweeps from f_min to f_max
        Phase 3: close[t] = base * (1 + Σᵢ Aᵢ * sin(fᵢ * t + φᵢ))
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps, dtype=np.float64)

    if phase == 1:
        # Fixed frequency sinusoid
        signal = amplitude_pct * np.sin(base_freq * t)

    elif phase == 2:
        # Chirp: frequency sweeps linearly from f_min to f_max
        f_min, f_max = freq_sweep_range
        # Instantaneous frequency: f(t) = f_min + (f_max - f_min) * t / n_steps
        # Phase: φ(t) = ∫f(τ)dτ = f_min*t + (f_max - f_min) * t²/(2*n_steps)
        phase_t = f_min * t + (f_max - f_min) * t**2 / (2 * n_steps)
        signal = amplitude_pct * np.sin(phase_t)

    elif phase == 3:
        # Fractal: sum of multiple sinusoids at different frequencies
        if fractal_freqs is None:
            fractal_freqs = [0.005, 0.02, 0.08, 0.2]
        if fractal_amplitudes is None:
            fractal_amplitudes = [0.06, 0.03, 0.015, 0.005]

        signal = np.zeros(n_steps, dtype=np.float64)
        random_phases = rng.uniform(0, 2 * np.pi, len(fractal_freqs))
        for freq, amp, phi in zip(fractal_freqs, fractal_amplitudes, random_phases):
            signal += amp * np.sin(freq * t + phi)

    else:
        raise ValueError(f"Invalid phase: {phase}. Must be 1, 2, or 3.")

    close = base_price * (1.0 + signal)
    return close


def synthetic_ohlc_from_close(
    close: np.ndarray,
    noise_pct: float = 0.002,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Generate synthetic OHLC from close array.

    Args:
        close: Close prices, shape (n_steps,)
        noise_pct: Noise amplitude as fraction of price (e.g., 0.002 = 0.2%)
        seed: Random seed

    Returns:
        (open, high, low, close) arrays, each shape (n_steps,)

    Logic:
        open[t] = close[t-1] + small noise (first open = close[0])
        high[t] = max(open[t], close[t]) * (1 + noise)
        low[t]  = min(open[t], close[t]) * (1 - noise)
    """
    rng = np.random.default_rng(seed)
    n = len(close)

    open_ = np.zeros(n, dtype=np.float64)
    open_[0] = close[0]
    open_[1:] = close[:-1] * (1 + rng.normal(0, noise_pct, n - 1))

    high_base = np.maximum(open_, close)
    high = high_base * (1 + rng.uniform(0, noise_pct, n))

    low_base = np.minimum(open_, close)
    low = low_base * (1 - rng.uniform(0, noise_pct, n))

    return open_, high, low, close


def _simple_moving_average(arr: np.ndarray, window: int) -> np.ndarray:
    """Compute simple moving average with forward-fill for leading NaNs."""
    if window <= 0:
        return arr
    n = len(arr)
    result = np.full(n, np.nan, dtype=np.float64)
    cumsum = np.cumsum(arr)
    result[window - 1:] = (cumsum[window - 1:] - np.concatenate([[0], cumsum[:n - window]])) / window
    # Forward fill NaNs
    mask = np.isnan(result)
    idx = np.where(~mask, np.arange(n), 0)
    np.maximum.accumulate(idx, out=idx)
    result[mask] = result[idx[mask]]
    return result


def _compute_rsi(close: np.ndarray, window: int = 14) -> np.ndarray:
    """Compute RSI (Relative Strength Index) normalized to [0, 1]."""
    delta = np.diff(close, prepend=close[0])
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)

    avg_gain = _simple_moving_average(gain, window)
    avg_loss = _simple_moving_average(loss, window)

    rs = np.divide(avg_gain, avg_loss, out=np.ones_like(avg_gain), where=avg_loss != 0)
    rsi = 1.0 - (1.0 / (1.0 + rs))
    return rsi


def _compute_stochastic(close: np.ndarray, high: np.ndarray, low: np.ndarray, window: int = 14) -> np.ndarray:
    """Compute Stochastic %K normalized to [0, 1]."""
    n = len(close)
    result = np.full(n, 0.5, dtype=np.float64)
    for i in range(window - 1, n):
        lowest = np.min(low[i - window + 1:i + 1])
        highest = np.max(high[i - window + 1:i + 1])
        if highest - lowest > 1e-8:
            result[i] = (close[i] - lowest) / (highest - lowest)
    return result


def derive_synthetic_features(
    close: np.ndarray,
    open_: np.ndarray,
    high: np.ndarray,
    low: np.ndarray,
    feature_cols: list[str],
) -> dict[str, np.ndarray]:
    """Compute all Norm_* feature columns from synthetic OHLC.

    Args:
        close, open_, high, low: OHLC arrays, shape (n_steps,)
        feature_cols: List of expected Norm_* column names (e.g., from dm.features)

    Returns:
        dict mapping feature_col_name -> np.ndarray(n_steps,), values in realistic ranges

    Strategy:
        Map each feature to its normalization group based on name patterns, then derive:
        - log_ratio: log(close / SMA(close, window))
        - oscillator: RSI, stochastic, price_position → values in [0, 1]
        - percentage: (high-low)/close, rolling distance → values / 100
        - passthrough: log-returns, drawdown, Bollinger %B
        - robust_scaled: z-score, rolling sharpe
        - binary: candle patterns, 0/1 flags
    """
    n = len(close)
    features = {}

    # Precompute common rolling statistics
    sma_10 = _simple_moving_average(close, 10)
    sma_20 = _simple_moving_average(close, 20)
    sma_50 = _simple_moving_average(close, 50)
    sma_200 = _simple_moving_average(close, 200)

    log_returns_1 = np.diff(np.log(close), prepend=np.log(close[0]))

    for col in feature_cols:
        col_lower = col.lower()

        # === log_ratio group (21 features) ===
        if 'wma_10' in col_lower or 'sma_10' in col_lower:
            features[col] = np.log(close / (sma_10 + 1e-8))
        elif 'wma_20' in col_lower or 'sma_20' in col_lower:
            features[col] = np.log(close / (sma_20 + 1e-8))
        elif 'wma_50' in col_lower or 'sma_50' in col_lower:
            features[col] = np.log(close / (sma_50 + 1e-8))
        elif 'wma_200' in col_lower or 'sma_200' in col_lower:
            features[col] = np.log(close / (sma_200 + 1e-8))
        elif 'psar' in col_lower:
            # Proxy: log(close / lagged_close(2))
            features[col] = np.log(close / (np.roll(close, 2) + 1e-8))
        elif 'ichimoku_conversion' in col_lower:
            sma_9 = _simple_moving_average(close, 9)
            features[col] = np.log(close / (sma_9 + 1e-8))
        elif 'ichimoku_base' in col_lower:
            sma_26 = _simple_moving_average(close, 26)
            features[col] = np.log(close / (sma_26 + 1e-8))
        elif 'ichimoku_span_a' in col_lower:
            sma_17 = _simple_moving_average(close, 17)
            features[col] = np.log(close / (sma_17 + 1e-8))
        elif 'ichimoku_span_b' in col_lower:
            sma_52 = _simple_moving_average(close, 52)
            features[col] = np.log(close / (sma_52 + 1e-8))
        elif 'pivot' in col_lower:
            # Pivot points: log(close / rolling_mean)
            window = 20 if 'r1' in col_lower or 's1' in col_lower else 40
            pivot = _simple_moving_average(close, window)
            features[col] = np.log(close / (pivot + 1e-8))
        elif 'macd' in col_lower and 'signal' not in col_lower and 'histogram' not in col_lower:
            sma_12 = _simple_moving_average(close, 12)
            sma_26 = _simple_moving_average(close, 26)
            macd = sma_12 - sma_26
            features[col] = np.log(1 + np.abs(macd) / (close + 1e-8)) * np.sign(macd)
        elif 'macd_signal' in col_lower:
            sma_12 = _simple_moving_average(close, 12)
            sma_26 = _simple_moving_average(close, 26)
            macd = sma_12 - sma_26
            macd_signal = _simple_moving_average(macd, 9)
            features[col] = np.log(1 + np.abs(macd_signal) / (close + 1e-8)) * np.sign(macd_signal)
        elif 'macd_histogram' in col_lower:
            features[col] = np.random.normal(0, 0.01, n)  # Small noise
        elif 'keltner' in col_lower:
            # Keltner %B: position within channel
            features[col] = np.clip(np.random.uniform(0.3, 0.7, n), 0, 1)
        elif 'volume' in col_lower:
            # Synthetic volume: small log-ratio noise
            features[col] = np.random.normal(0, 0.05, n)
        elif 'mom_10' in col_lower or 'momentum_10' in col_lower:
            features[col] = np.log(close / (np.roll(close, 10) + 1e-8))
        elif 'mom_20' in col_lower or 'momentum_20' in col_lower:
            features[col] = np.log(close / (np.roll(close, 20) + 1e-8))

        # === oscillator group (9 features) ===
        elif 'rsi' in col_lower and 'norm_rsi' in col:
            features[col] = _compute_rsi(close, 14)
        elif 'stoch_k' in col_lower:
            features[col] = _compute_stochastic(close, high, low, 14)
        elif 'stoch_d' in col_lower:
            stoch_k = _compute_stochastic(close, high, low, 14)
            features[col] = _simple_moving_average(stoch_k, 3)
        elif 'plus_di' in col_lower or 'minus_di' in col_lower:
            features[col] = np.random.uniform(0.2, 0.8, n)
        elif 'mfi' in col_lower:
            features[col] = _compute_rsi(close, 14)  # Proxy with RSI
        elif 'price_position' in col_lower:
            # Price position in rolling window
            window = 10 if '10' in col_lower else (20 if '20' in col_lower else 50)
            result = np.full(n, 0.5, dtype=np.float64)
            for i in range(window - 1, n):
                low_w = np.min(close[i - window + 1:i + 1])
                high_w = np.max(close[i - window + 1:i + 1])
                if high_w - low_w > 1e-8:
                    result[i] = (close[i] - low_w) / (high_w - low_w)
            features[col] = result

        # === percentage group (10 features) ===
        elif 'donchian_width_pct' in col_lower:
            features[col] = (high - low) / (close + 1e-8)
        elif 'candle_body_pct' in col_lower:
            features[col] = np.abs(close - open_) / (close + 1e-8)
        elif 'candle_range_pct' in col_lower:
            features[col] = (high - low) / (close + 1e-8)
        elif 'pivot_distance_pct' in col_lower:
            features[col] = (close - sma_20) / (sma_20 + 1e-8)
        elif 'dist_from_high' in col_lower or 'dist_from_low' in col_lower:
            window = 10 if '10' in col_lower else (20 if '20' in col_lower else 50)
            roll_high = np.array([np.max(close[max(0, i - window + 1):i + 1]) for i in range(n)])
            roll_low = np.array([np.min(close[max(0, i - window + 1):i + 1]) for i in range(n)])
            if 'dist_from_high' in col_lower:
                features[col] = (close - roll_high) / (roll_high + 1e-8)
            else:
                features[col] = (close - roll_low) / (roll_low + 1e-8)

        # === passthrough group (7 features) ===
        elif 'log_returns_1' in col_lower:
            features[col] = log_returns_1
        elif 'log_returns_5' in col_lower:
            features[col] = np.log(close / (np.roll(close, 5) + 1e-8))
        elif 'log_returns_10' in col_lower:
            features[col] = np.log(close / (np.roll(close, 10) + 1e-8))
        elif 'bb_percent_b' in col_lower or 'bb_bandwidth' in col_lower:
            # Bollinger bands: (close - SMA) / (2*std)
            std_20 = np.array([np.std(close[max(0, i - 19):i + 1]) for i in range(n)])
            features[col] = (close - sma_20) / (2 * std_20 + 1e-8)
        elif 'cmf' in col_lower:
            features[col] = np.random.uniform(-0.3, 0.3, n)
        elif col_lower.endswith('drawdown') and 'drawdown_pct' not in col_lower:
            # Running drawdown
            cummax = np.maximum.accumulate(close)
            features[col] = (close - cummax) / (cummax + 1e-8)

        # === robust_scaled group (6 features) ===
        elif 'zscore' in col_lower:
            features[col] = (close - sma_20) / (np.std(close) + 1e-8)
        elif 'sharpe_ratio' in col_lower:
            roll_ret = np.diff(np.log(close), prepend=np.log(close[0]))
            roll_sharpe = roll_ret / (np.std(roll_ret) + 1e-8)
            features[col] = _simple_moving_average(roll_sharpe, 20)
        elif 'drawdown_pct' in col_lower:
            cummax = np.maximum.accumulate(close)
            features[col] = (close - cummax) / (cummax + 1e-8) * 100
        elif 'ad_line' in col_lower:
            features[col] = np.cumsum(log_returns_1)
        elif 'trend_strength' in col_lower:
            features[col] = (close - sma_50) / (sma_50 + 1e-8)
        elif 'cumulative_returns' in col_lower:
            features[col] = np.cumsum(log_returns_1)

        # === binary group (12 features) ===
        elif 'candle_body_ratio' in col_lower:
            features[col] = np.abs(close - open_) / ((high - low) + 1e-8)
        elif 'candle_upper_wick_ratio' in col_lower:
            features[col] = (high - np.maximum(open_, close)) / ((high - low) + 1e-8)
        elif 'candle_lower_wick_ratio' in col_lower:
            features[col] = (np.minimum(open_, close) - low) / ((high - low) + 1e-8)
        elif 'candle_direction' in col_lower:
            features[col] = (close > open_).astype(np.float64)
        elif 'pattern_doji' in col_lower:
            features[col] = (np.abs(close - open_) < 0.001 * close).astype(np.float64)
        elif 'pattern_hammer' in col_lower or 'pattern_inverted_hammer' in col_lower:
            features[col] = np.random.choice([0.0, 1.0], n, p=[0.95, 0.05])
        elif 'pattern_marubozu' in col_lower or 'pattern_spinning_top' in col_lower:
            features[col] = np.random.choice([0.0, 1.0], n, p=[0.97, 0.03])
        elif 'pattern_engulfing' in col_lower or 'pattern_strong_momentum' in col_lower:
            features[col] = np.random.choice([0.0, 1.0], n, p=[0.9, 0.1])

        # === Fallback ===
        else:
            # Default: small random noise centered at 0
            features[col] = np.random.normal(0, 0.1, n)

    # Clip all features to [-5, 5] and sanitize
    for col in features:
        features[col] = np.clip(features[col], -5, 5)
        features[col] = np.nan_to_num(features[col], nan=0.0, posinf=5.0, neginf=-5.0)

    return features


def make_synthetic_df(
    n_steps: int,
    phase: int,
    feature_cols: list[str],
    base_price: float = 50_000.0,
    amplitude_pct: float = 0.10,
    base_freq: float = 0.02,
    noise_pct: float = 0.002,
    seed: int = 42,
    **phase_kwargs,
) -> pl.DataFrame:
    """Create a complete synthetic Polars DataFrame consumable by CryptoMarketEnv.

    Args:
        n_steps: Number of timesteps
        phase: 1, 2, or 3 (sinusoid type)
        feature_cols: List of expected Norm_* column names (e.g., dm.features)
        base_price: Base price level
        amplitude_pct: Oscillation amplitude
        base_freq: Base frequency (phase 1)
        noise_pct: OHLC noise level
        seed: Random seed
        **phase_kwargs: Additional args for generate_synthetic_prices
            (freq_sweep_range, fractal_freqs, fractal_amplitudes)

    Returns:
        pl.DataFrame with columns: open, high, low, close + all feature_cols

    Example:
        >>> from src.data.manager import DataManager
        >>> dm = DataManager(interval="60m")
        >>> df_synth = make_synthetic_df(10_000, phase=1, feature_cols=dm.features)
        >>> env = CryptoMarketEnvConsolidatedReward(df_synth, dm.features)
    """
    # Generate prices
    close = generate_synthetic_prices(
        n_steps, phase, base_price, amplitude_pct, base_freq,
        seed=seed, **phase_kwargs
    )

    # Derive OHLC
    open_, high, low, close = synthetic_ohlc_from_close(close, noise_pct, seed)

    # Derive features
    features = derive_synthetic_features(close, open_, high, low, feature_cols)

    # Build DataFrame
    data = {
        "open": open_,
        "high": high,
        "low": low,
        "close": close,
    }
    data.update(features)

    return pl.DataFrame(data)
